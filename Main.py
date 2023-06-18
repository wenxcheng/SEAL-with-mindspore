import mindspore as ms
from mindspore import nn
import numpy as np
import sys, copy, math, time, pdb #pdb:python自带的代码调试包
import pickle #直接将python对象写入二进制文件
import scipy.io as sio #解决不同格式文件的输入和输出
import scipy.sparse as ssp #用于初始化和操作稀疏矩阵
import os.path #主要用于获取文件的属性
import random
import argparse #用于解析命令行参数和选项
#from torch.utils.data import DataLoader--ms加载数据集？？
sys.path.append('%s/../pytorch_DGCNN' % os.path.dirname(os.path.realpath(__file__))) #添加自定义模块搜索目录,引用模块和执行脚本不在一个目录
from main import *
from util_functions import *

parser = argparse.ArgumentParser(description='Link Prediction with SEAL')
#general settings （带--为可选参数
parser.add_argument('--data-name', default=None, help='network name')
parser.add_argument('--train-name', default=None, help='train name')
parser.add_argument('--test-name', default=None, help='test name')
parser.add_argument('--only-predict', action='store_true', default=False,
                    help='if True, will load the saved model and output predicitons for links in test-name;'
                         'you still need to spcify train-name in order to build the observed network and extract subgraphs')
parser.add_argument('--batch-size', type=int, default=50)
parser.add_argument('--max-train-num', type=int, default=10000,
                    help='set maximum number of train links(to fit into memory)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed(default=1)')
parser.add_argument('--test-ratio', type=float, default=0.1, help='raion of test links')
parser.add_argument('--no-parallel', action='store_true', default=False,
                    help='if True, use single thread for subgraph extraction; by default use all cpu cores to extract subgraphs in parallel')
parser.add_argument('--all-unknown-as-negative', action='store_true', default=False,
                    help='if True, regard all unknown links as negative test data; sample a portion from them as negative training data.'
                         'Otherwise, train negative and test negative are both sampled from unknown links without overlap')
#model setting
parser.add_argument('--hop', default=1, metavar='S',help='enclosing subgraph hop number, options:1,2,..,auto') #metavar 用于指定参数名称
parser.add_argument('--max-nodes-per-hop', default=None, help='if>0, upper bound the number per hop by subsampling')
parser.add_argument('--use-embedding', action='store_true', default=False, help='whether to use node2vec node embeddings')
parser.add_argument('--use-attribute', action='store_true', default=False, help='whether to use node attributes')
parser.add_argument('--save-model', action='store_true', default=False, help='save final model')
args = parser.parse_args() #将parser中设置的所有参数返回给args实例
ms.set_seed(args.seed) #生成种子
# if args.cuda:
#     torch.cuda.manual_seed(args.seed)
print(args)

#固定三方的随机数种子。（cmd_args哪来的？
random.seed(cmd_args.seed) #python的
np.random.seed(cmd_args.seed) #第三方库Numpy的
ms.set_seed(cmd_args.seed)
if args.hop != 'auto':
    args.hop = int(args.hop)
if args.max_nodes_per_hop is not None:
    args.max_nodes_per_hop = int(max_nodes_per_hop)


'''Prepare data'''
args.file_dir = os.path.dirname(os.path.realpath('__file__'))

#检查训练和测试连接是否准备好
train_pos, test_pos = None, None
if args.train_name is not None:
    args.train_dir = os.path.join(args.file_dir, 'data/{}'.format(args.train_name))
    train_idx = np.loadtxt(args.train_dir, dtype=int)
    train_pos = (train_idx[:, 0], train_idx[:, 1])
if args.test_name is not None:
    args.test_dir = os.path.join(args.file_dir, 'data/{}'.format(args.test_name))
    test_idx = np.loadtxt(args.test_dir, dtype=int)
    test_pos = (test_idx[:, 0], test_idx[:, 1])

#建立观察到的网络
if args.data_name is not None: #use .mat network
    args.data_dir = os.path.join(args.file_dir, 'data/{}.mat'.format(args.data_name))
    data = sio.loadmat(args.data_dir)
    net = data['net']
    if 'group' in data:
        #加载节点属性
        attributes = data['group'].toarray().astype('float32')
    else:
        attributes = None
    #查看网络是否对称（只针对小网络）
    if False:
        net_ = net.toarray()
        assert(np.allclose(net_, net_.T, atol=1e-8))
else: #(无指定数据集?)从训练连接中建立网络
    assert (args.train_name is not None), "must provide train links if not using .mat"
    if args.train_name.endswith('_train.txt'):
        args.data_name = args.train_name[:-10]
    else:
        args.data_name = args.train_name.split('.')[0]
    max_idx = np.max(train_idx)
    if args.test_name is not None:
        max_idx = max(max_idx, np.max(test_idx))
    net = ssp.csc_matrix(
        (np.ones(len(train_idx)), (train_idx[:, 0], train_idx[:, 1])),
        shape=(max_idx+1, max_idx+1)
    )
    net[train_idx[:, 1], train_idx[:,0]] = 1 #增加对称边
    net[np.arange(max_idx+1), np.arange(max_idx+1)] = 0 #移除自环


#采样训练和测试连接
if args.train_name is None and args.test_name is None:
    #从网络中同事采样正向和负向的训练/测试连接
    train_pos, train_neg, test_pos, test_neg = sample_neg(
        net, args.test_ratio, max_train_num=args.max_train_num
    )
else:
    #使用已有的训练/测试正向链接，从网络中采样负连接
    train_pos, train_neg, test_pos, test_neg = sample_neg(
        net,
        train_pos=train_pos,
        test_pos=test_pos,
        max_train_num=args.max_train_num,
        all_unknown_as_negative=args.all_unkown_as_negative
    )

'''训练并应用分类器'''
A = net.copy()
A[test_pos[0], test_pos[1]] = 0 #mask test links
A[test_pos[1], test_pos[0]] = 0 #mask test links
A.eliminate_zeros() #使用scipy-1.3.x中的稀疏矩阵时，需要连接是遮蔽？的

node_information = None
if args.use_embedding:
    embeddings = generate_node2vec_embeddings(A, 128, True, train_neg)
    node_information = embeddings
if args.use_attribute and attibutes is not None:
    if node_information is not None:
        node_information = np.concatnate([node_information, attributes], axis=1)
    else:
        node_information = attributes
if args.only_predict: #无需使用负连接
    _, test_graphs, max_n_label = links2subgraphs(
        A,
        None,
        None,
        test_pos, #只是一个名称，我们实际并不知道他们的标签
        None,
        args.hop,
        args.max_nodes_per_hop,
        node_information,
        args.no_parallel
    )
    print('# test: %d' % (len(test_graphs)))
else:
    train_graphs, test_graphs, max_n_label = links2subgraphs(
        A,
        train_pos,
        train_neg,
        test_pos,
        test_neg,
        args.hop,
        args.max_nodes_per_hop,
        node_information,
        args.no_parallel
    )
    print('# train: %d, # test: %d' % len(train_graphs), len(test_graphs))


#配置DGCNN
if args.only_predict:
    with open('data/{}_hyper.pkl'.format(args.data_name), 'rb') as hyperparameters_name:
        saved_cmd_args = pickle.load(hyperparameters_name)
        for key, value in vars(saved_cmd_args).items(): #用cmd_args代替
            vars(cmd_args)[key] = value
        classifier = Classifier()
        # if  cmd_args.mode == 'gpu':
        #     classifier = classifier.cuda()
        #model_name = 'data/{}_model.pth'.format(args.data_name)
        model_name = 'data/{}_model.ckpt'.format(args.data_name)
        #classifier.load_state_dict(torch.load(model_name))
        ms.load_param_into_net(classifier, model_name)
        #classifier.eval() #类似train()但用在无DN和Dropout的test中
        classifier.set_train(False)
        predictions = []
        batch_graph = []
        for i, graph in enumerate(test_graphs):
            batch_graph.append(graph)
            if len(batch_graph) == cmd_args.batch_size or i == (len(test_graphs)-1):
                predictions.append(classifier(batch_graph)[0][:, 1].exp().cpu().detach()) #返回指数，变量放在cpu上，阻断反向传播，返回值仍为tensor(pytorch函数)
                batch_graph = []
        #predictions = torch.cat(predictions, 0).unsqueeze(1).numpy() #沿0维度(纵向)拼接，沿1维升维
        predictions = ms.concat(predictions).numpy()
        test_idx_and_pred = np.concatnate([test_idx, predictions], 1)
        pred_name = 'data/' + args.test_name.split('.')[0] + '_pred.txt'
        np.savetxt(pred_name, test_idx_and_pred, fmt=['%d' ,'%d', '%1.2f'])
        print('Predicitions for {} are saved in {}'.format(args.test_name, pred_name))
        exit()


cmd_args.gm = 'DGCNN'
cmd_args.sortpooling_k = 0.6
cmd_args.latent_dim = [32, 32, 32, 1]
cmd_args.hidden = 128
cmd_args.out_dim = 0
cmd_args.dropout = True
cmd_args.num_class = 2
# cmd_args.mode = 'gpu' if args.cuda else 'cpu'
cmd_args.mode = 'cpu'
cmd_args.num_epochs = 50
cmd_args.learning_rate = 1e-4
cmd_args.printAUC = True
cmd_args.feat_dim = max_n_label + 1
cmd_args.attr_dim = 0
if node_information is not None:
    cmd_args.attr_dim = node_information.shape[1]
if cmd_args.sortpooling_k <= 1:
    num_nodes_list = sorted([g.num_nodes for g in train_graphs + test_graphs])
    k_ = int(math.ceil(cmd_args.sortpooling_k * len(num_nodes_list))) - 1
    cmd_args.sortpooling_k = max(10, num_nodes_list[k_])
    print('k used in SortPooling is :' + str(cmd_args.sortpooling_k))

classifier = Classifier()
# if cmd_args.mode == 'gpu':
#     classifier = classifier.cuda()

optimizer = nn.Adam(classifier.parameters(), learning_rate=cmd_args.learning_rate)

random.shuffle(train_graphs)
val_num = int(0.1 * len(train_graphs))
val_graphs = train_graphs[:val_num]
train_graphs = train_graphs[val_num:]

train_idxes = list(range(len(train_graphs)))
best_loss = None
best_epoch = None
for epoch in range(cmd_args.num_epochs):
    random.shuffle(train_idxes)
    classifier.set_train() #默认true的训练模式
    avg_loss = loop_dataset(
        train_graphs, classifier, train_idxes, optimizer=optimizer, bsize=args.batch_size
    )
    if not cmd_args.printAUC:
        avg_loss[2] = 0.0
    print('\033[92maverage training of epoch %d: loss %.5f acc %.5f auc %.5f\033[0m' % (
        epoch, avg_loss[0], avg_loss[1], avg_loss[2]
    )) #\033成对出现：ANSI控制码，92——绿色，[0m——关闭所有属性

    classifier.set_train(False)
    val_loss = loop_dataset(val_graphs, classifier, list(range(len(val_graphs))))
    if not cmd_args.printAUC:
        val_loss[2] = 0.0
    print('\033[92maverage validation of epoch %d: loss %.5f acc %.5f auc %.5f\033[0m' % (
            epoch, val_loss[0], val_loss[1], val_loss[2]
    ))
    if best_loss is not None:
        best_loss = val_loss
    if val_loss[0] <= best_loss[0]:
        best_loss = val_loss
        best_epoch = epoch
        test_loss = loop_dataset(test_graphs, classifier, list(range(len(test_graphs))))
        if not cmd_args.printAUC:
            test_loss[2] = 0.0
        print('\033[92maverage validation of epoch %d: loss %.5f acc %.5f auc %.5f\033[0m' % (
            epoch, test_loss[0], test_loss[1], test_loss[2]
        ))

print('\033[92maverage validation of epoch %d: loss %.5f acc %.5f auc %.5f\033[0m' % (
            epoch, test_loss[0], test_loss[1], test_loss[2]
))

if args.save_model:
    model_name = 'data/{}_model.ckpt'.format(args.data_name)
    print('Saving final model states to {}...'.format(model_name))
    # torch.save(classifier.state_dict(), model_name) #只保存权重
    ms.save_checkpoint(classifier, model_name)
    hyper_name = 'data/{}_hyper.ckpt'.format(args.data_name)
    with open(hyper_name, 'wb') as hyperparameters_file:
        pickle.dump(cmd_args, hyperparameters_file)
        print('Saving hyperparameters states to {}...'.format(hyper_name))

with open('acc_results.txt','a+') as f:
    f.write(str(test_loss[1]) + '\n')

if cmd_args.printAUC:
    with open('auc_results.txt', 'a+') as f:
        f.write(str(test_loss[2]) + '\n')
