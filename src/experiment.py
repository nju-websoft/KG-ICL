import torch

torch.cuda.current_device()
import time
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from encoder.EntityEncoder import *
from data_loader import *
from utils import cal_ranks, cal_performance
import os, random
from tqdm import tqdm
from torch.nn import init
import sys


class Experiment(object):
    def __init__(self, args):
        self.args = args

        # 1. prepare data
        if args.train_dirs is not None:
            self.train_loaders = [DataLoader(args, train_dir, train_dir.split('/')[-2]) for train_dir in
                                  args.train_dirs]
        else:
            self.train_loaders = None
        if args.valid_dirs is not None:
            self.valid_loaders = [DataLoader(args, valid_dir, valid_dir.split('/')[-2]) for valid_dir in
                                  args.valid_dirs]
        else:
            self.valid_loaders = None

        # 2. initialize model
        self.model = EntityEncoder(self.args)
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                try:
                    init.xavier_uniform_(param.data)
                except:
                    pass
            elif 'bias' in name:
                init.constant_(param.data, 0)
            print(name, param.shape, param.numel())
        self.num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.args.logger.info('Number of parameters: {}'.format(self.num_params))

        self.model.to(self.args.device)

        self.optimizer = Adam(self.model.parameters(), lr=args.lr, weight_decay=args.lamb)
        self.scheduler = ExponentialLR(self.optimizer, args.decay_rate)
        self.t_time = 0

    def load_test(self):
        self.test_loaders = [DataLoader(self.args, test_dir, test_dir.split('/')[-2]) for test_dir in
                                 self.args.test_dirs]

    def save_model(self, is_best=False, epoch=0, save_path=''):
        '''
        Save trained model.
        :param is_best: If True, save it as the best model.
        After training on each snapshot, we will use the best model to evaluate.
        '''
        checkpoint_dict = dict()
        checkpoint_dict['state_dict'] = self.model.state_dict()
        checkpoint_dict['optimizer_state_dict'] = self.optimizer.state_dict()
        checkpoint_dict['epoch_id'] = epoch

        if is_best:
            print('Saving Best Model to {}/model_best.tar'.format(save_path))
            out_tar = os.path.join(save_path, 'model_best.tar')
            torch.save(checkpoint_dict, out_tar)

    def load_checkpoint(self, input_file, gpu):
        '''
        :param input_file:  path to the checkpoint
        :param gpu:  gpu id
        :return:  load the model from the checkpoint
        '''
        input_file = os.path.join(input_file, 'model_best.tar')
        if os.path.isfile(os.path.join(os.getcwd(), input_file)):
            print('=> loading checkpoint \'{}\''.format(os.path.join(os.getcwd(), input_file)))
            if torch.cuda.is_available():
                checkpoint = torch.load(os.path.join(os.getcwd(), input_file), map_location="cuda:{}".format(gpu))
            else:
                checkpoint = torch.load(os.path.join(os.getcwd(), input_file), map_location="cpu")
            for key in list(checkpoint['state_dict'].keys()):
                if key not in list(self.model.state_dict().keys()):
                    del checkpoint['state_dict'][key]
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            print('=> no checkpoint found at \'{}\''.format(input_file))

    def train_batch(self, finetune=False, step_num_list=None, max_step=1000):
        epoch_loss = 0
        success = False
        now_loader = None
        while not success:
            try:  # if the memory is not enough, reduce the batch size
                for train_loader in self.train_loaders:
                    # shuffle training data
                    train_loader.kg.data = train_loader.kg.data[np.random.permutation(len(train_loader.kg.data))]

                if step_num_list is None:  # you can set the number of steps for each dataset
                    n_batchs = [len(train_loader.kg.data) // train_loader.kg.train_batch_size + (
                        1 if (len(train_loader.kg.data) % train_loader.kg.train_batch_size) > 0 else 0) for train_loader
                                in self.train_loaders]

                    train_batchs = []
                    for loader_id, batches in enumerate(n_batchs):
                        for batch_id in range(batches):
                            train_batchs.append([loader_id, batch_id])
                else:
                    n_batchs = [len(train_loader.kg.data) // train_loader.kg.train_batch_size + (
                        1 if (len(train_loader.kg.data) % train_loader.kg.train_batch_size) > 0 else 0) for train_loader
                                in self.train_loaders]
                    train_batchs = []
                    for loader_id, batches in enumerate(n_batchs):
                        for batch_id in range(batches):
                            train_batchs.append([loader_id, batch_id])

                random.shuffle(train_batchs)

                t_time = time.time()
                self.model.train()
                count_step = 0
                for batch in tqdm(train_batchs, ncols=100, position=0, leave=True):
                    count_step += 1
                    if max_step is not None and count_step > max_step:
                        break
                    self.model.zero_grad()
                    j, i = batch
                    now_loader = self.train_loaders[j]
                    triple = self.train_loaders[j].kg.data[i * now_loader.kg.train_batch_size:min(len(now_loader.kg.data), (i + 1) * now_loader.kg.train_batch_size)]
                    scores, rel_embeddings_full, rel_labels = self.model(triple[:, 0], triple[:, 1], triple[:, 2], loader=self.train_loaders[j], training=True, finetune=finetune)
                    pos_scores = scores[[torch.arange(len(scores)).to(self.args.device), torch.LongTensor(triple[:, 2]).to(self.args.device)]]
                    # delete invalid samples
                    valid_samples = torch.nonzero(pos_scores != 0).squeeze()
                    scores = scores[valid_samples]
                    pos_scores = pos_scores[valid_samples]
                    if len(scores.shape) == 1:
                        scores = scores.unsqueeze(0)
                    max_n = torch.max(scores, 1, keepdim=True)[0]
                    loss = torch.mean(- pos_scores + max_n + torch.log(torch.sum(torch.exp(scores - max_n), 1)))
                    loss.backward()
                    self.optimizer.step()

                    # avoid NaN
                    for p in self.model.parameters():
                        X = p.data.clone()
                        flag = X != X
                        X[flag] = np.random.random()
                        p.data.copy_(X)
                    epoch_loss += loss.item()
                self.scheduler.step()
                self.t_time += time.time() - t_time
                if finetune:  # if finetune, we do not need to validate the model
                    valid_mrr, out_str = 0, 0
                else:
                    valid_mrr, out_str = self.evaluate(mode='valid')
                success = True
            except:
                if 'CUDA out of memory' in str(sys.exc_info()[1]):
                    print('CUDA out of memory, try to reduce batch size')
                    now_loader.kg.train_batch_size = now_loader.kg.train_batch_size // 2
                    print('batch size reduced to', now_loader.kg.train_batch_size)
                else:
                    print('Unexpected error:', sys.exc_info()[0])
                    raise
                torch.cuda.empty_cache()
        return valid_mrr, out_str, epoch_loss

    def evaluate(self, mode='test'):
        if mode == 'valid':
            loaders = self.valid_loaders
        elif mode == 'test':
            loaders = self.test_loaders
        elif mode == 'train':
            loaders = self.train_loaders
        else:
            raise ('evaluation mode error')

        v_mrr_list = []
        out_str_list = []
        with torch.no_grad():
            for loader in loaders:
                success = False
                while not success:
                    try:
                        order = list(range(len(loader.kg.query)))
                        random.shuffle(order)
                        loader.kg.query = [loader.kg.query[i] for i in order]
                        loader.kg.answer = [loader.kg.answer[i] for i in order]

                        n_data = len(loader.kg.query)
                        n_batch = n_data // loader.kg.test_batch_size + int(n_data % loader.kg.test_batch_size > 0)
                        ranking = []
                        self.model.eval()
                        i_time = time.time()
                        for i in tqdm(range(n_batch), ncols=100, position=0, leave=True):
                            start = i * loader.kg.test_batch_size
                            end = min(n_data, (i + 1) * loader.kg.test_batch_size)
                            batch_idx = np.arange(start, end)
                            subs, rels, objs = loader.get_batch(batch_idx, mode=mode)
                            scores = self.model(subs, rels, loader=loader, training=False)[0].data.cpu().numpy()
                            # filter out the known positive triples
                            filters = []
                            for i in range(len(subs)):
                                filt = loader.kg.filter[(subs[i], rels[i])]
                                filt_1hot = np.zeros((loader.kg.entity_num,))
                                filt_1hot[np.array(filt)] = 1
                                filters.append(filt_1hot)
                            filters = np.array(filters)
                            ranks = cal_ranks(scores, objs, filters)
                            ranking += list(ranks)
                        ranking = np.array(ranking)
                        v_mrr, v_h1, v_h3, v_h5, v_h10 = cal_performance(ranking)

                        # prepare logging
                        i_time = time.time() - i_time
                        self.args.logger.info(loader.name+' MRR:%.4f H@1:%.4f H@3:%.4f H@5:%.4f H@10:%.4f[TIME] train:%.4f inference:%.4f\n' % (
                            v_mrr, v_h1, v_h3, v_h5, v_h10, self.t_time, i_time))
                        print(loader.name+' MRR:%.4f H@1:%.4f H@3:%.4f H@5:%.4f H@10:%.4f[TIME] train:%.4f inference:%.4f\n' % (
                            v_mrr, v_h1, v_h3, v_h5, v_h10, self.t_time, i_time))
                        out_str = loader.name + ' MRR:%.4f H@1:%.4f H@3:%.4f H@5:%.4f H@10:%.4f[TIME] train:%.4f inference:%.4f\n' % (
                            v_mrr, v_h1, v_h3, v_h5, v_h10, self.t_time, i_time)
                        v_mrr_list.append(v_mrr)
                        out_str_list.append(out_str)
                        success = True
                    except:
                        if 'CUDA out of memory' in str(sys.exc_info()[1]):
                            print('CUDA out of memory, try to reduce batch size')
                            loader.kg.test_batch_size = loader.kg.test_batch_size // 2
                            print('batch size reduced to', loader.kg.test_batch_size)
                        else:
                            print('Unexpected error:', sys.exc_info()[0])
                            raise

        v_mrr = np.mean(v_mrr_list)
        out_str = ''.join(out_str_list)
        return v_mrr, out_str
