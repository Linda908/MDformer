import datetime

from getData import *
from model import *


def fold_valid(args):
    similarity_feature = similarity_feature_process(args)
    edge_idx_dict, g = load_fold_data(args)
    n_rna = edge_idx_dict['true_md'].shape[0]
    n_dis = edge_idx_dict['true_md'].shape[1]

    model = MY_Module(args, n_rna, n_dis).to(args.device)
    print(model)
    print("*******************************************************************")
    metric_result_list = []
    metric_result_list_str = []
    metric_result_list_str.append('AUC    AUPR    Acc    F1    pre    recall')
    for i in range(args.kfolds):
        model = MY_Module(args, n_rna, n_dis).to(args.device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
        criterion = torch.nn.BCEWithLogitsLoss().to(args.device)

        print(f'###########################Fold {i + 1} of {args.kfolds}###########################')
        Record_res = []
        Record_res.append('AUC    AUPR    Acc    F1    pre    recall')
        model.train()
        for epoch in range(args.epoch):
            optimizer.zero_grad()

            out = model(args, similarity_feature, g, edge_idx_dict, edge_idx_dict[str(i)]['fold_train_edges_80p_80n'],
                        i).view(-1)

            loss = criterion(out, edge_idx_dict[str(i)]['fold_train_label_80p_80n'])
            loss.backward()
            optimizer.step()

            test_auc, metric_result, y_true, y_score = valid_fold(args, model,
                                                                  similarity_feature,
                                                                  g,
                                                                  edge_idx_dict,
                                                                  edge_idx_dict[str(i)]['fold_valid_edges_20p_20n'],
                                                                  i
                                                                  )
            One_epoch_metric = '{:.4f}    {:.4f}    {:.4f}    {:.4f}    {:.4f}    {:.4f} '.format(*metric_result)
            Record_res.append(One_epoch_metric)
            if epoch + 1 == args.epoch:
                metric_result_list.append(metric_result)
                metric_result_list_str.append(One_epoch_metric)
            print('epoch {:03d} train_loss {:.8f} val_auc {:.4f} '.format(epoch + 1, loss.item(), test_auc))

    arr = np.array(metric_result_list)
    averages = np.round(np.mean(arr, axis=0), 4)
    metric_result_list_str.append('平均值：')
    metric_result_list_str.append('{:.4f}    {:.4f}    {:.4f}    {:.4f}    {:.4f}    {:.4f} '.format(*list(averages)))


    now = datetime.datetime.now()
    with open('平均_' + now.strftime("%Y_%m_%d_%H_%M_%S") + '_.txt', 'w') as f:
        f.write('\n'.join(metric_result_list_str))
    return averages


def valid_fold(args, model, similarity_feature, graph, edge_idx_dict, edge_label_index, i):
    lable = edge_idx_dict[str(i)]['fold_valid_label_20p_20n']

    model.eval()
    with torch.no_grad():
        out = model.encode(args, similarity_feature, graph, edge_idx_dict, i)
        res = model.decode(out, edge_label_index, i).view(-1).sigmoid()
        model.train()
    metric_result = caculate_metrics(lable.cpu().numpy(), res.cpu().numpy())
    my_acu = metrics.roc_auc_score(lable.cpu().numpy(), res.cpu().numpy())
    return my_acu, metric_result, lable, res
