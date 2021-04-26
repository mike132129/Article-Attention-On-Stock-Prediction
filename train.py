import torch
<<<<<<< HEAD
import 
=======
import torch.nn as nn

from transformers import BertTokenizer

from dataset import news_dataset
from model import StockModel

def main(args):
    trans = transforms.Compose([
        transforms.ToTensor(),
        ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_version = 'bert-base-chinese'
	tokenizer = BertTokenizer.from_pretrained(model_version)
    train_dataset =  news_dataset(tokenizer)
    val_dataset = news_dataset(tokenizer)

    print('Train loader length: {}'.format(len(train_loader)))

    #model
    model = StockModel(13)
    model.to(device)

    # loss function
    sample_loss_function = nn.PoissonNLLLoss()

    # optimize
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= int(args.num_epochs / 3), gamma=0.5)

    # tensorboard
    writer = SummaryWriter()

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        total_loss= 0
        
        for index, data in enumerate(train_loader, 0):
            train_x, train_y = data
            pdb.set_trace()
            model.zero_grad()
            
            train_x = train_x.to(device)
            train_y = train_y.to(device)
            stats_prop = model(train_x)

            # pdb.set_trace()

            # a little weighted
            loss = stat_loss_function(train_y, stats_prop)

            # print(loss)
            total_loss+= loss.item()
            loss.backward()
            optimizer.step()
            if index % args.print_step == 0 or index == len(train_loader) - 1:
                print('Epoch {} [{} / {} ({:.2f} %)]: loss: {}'.format(epoch, index, len(train_loader), (index / len(train_loader) * 100), total_loss / (index + 1)), flush=True)

        scheduler.step()
        pre_y_array = None

        #keep saving the same model
        save_model(model, args.model_dir, 0)

        # testing
        y_pred = []
        y_test = []
        model.eval()
        test_loss = 0.0
        for tIndex, data in enumerate(val_loader):
            test_x , test_y = data

            test_x = test_x.to(device)
            test_y = test_y.to(device)

            stats_prop = model(test_x)


            # a little weighted
            loss = stat_loss_function(stats_prop, test_y)
            
            # store the result
            y_pred.append(stats_prop.detach().cpu())
            y_test.append(test_y.detach().cpu())

            # print(loss)
            if torch.isnan(loss):
                continue
            test_loss += loss.item()

        print("Train loss: {}, Test loss: {} ".format(total_loss / len(train_loader), test_loss / len(val_loader)))
        # draw on tensorboard

        writer.add_scalar('Train/loss_{}'.format(args.task_name), total_loss / len(train_loader), epoch)
        writer.add_scalar('Test/loss_{}'.format(args.task_name), test_loss / len(val_loader), epoch)

        if epoch % args.store_model == 0 or epoch == 1:
            save_model(model, args.model_dir, epoch)

        # export prediction result
        if epoch % args.store_epoch == 0 or epoch == 1:
            if not os.path.isdir('./test_data/csv/{}'.format(args.task_name)):
                Path('./test_data/csv/{}'.format(args.task_name)).mkdir(parents=True, exist_ok=True)
                Path('./test_data/fig/{}'.format(args.task_name)).mkdir(parents=True, exist_ok=True)
            y_test = torch.cat(y_test, dim=1)
            y_test = pd.DataFrame(y_test.detach().cpu().numpy().reshape(-1, 13))
            y_test.to_csv('./test_data/csv/{}/rainfall_test_y.csv'.format(args.task_name))

            y_pred = torch.cat(y_pred, dim=1)
            y_pred = pd.DataFrame(y_pred.detach().cpu().numpy().reshape(-1, 13))
            y_pred.to_csv('./test_data/csv/{}/rainfall_pred_{}_y.csv'.format(args.task_name, epoch))
            # pdb.set_trace()
            # final.to_csv('./test_data/pure_rain_pred_{}_y.csv'.format(epoch))
            # obs_path = './test_data/rainfall_test_y.csv'
            # pre_path = './test_data/pure_rain_pred_{}_y.csv'.format(epoch)
            save_path =  './test_data/fig/{}/result_fig_{}_smoothl1loss.png'.format(args.task_name, epoch)
            visualize_store(y_test, y_pred, save_path)

if __name__ == '__main__':
    file_path = 'configs/default.yaml'
    
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))
    
    args = DotMap(yaml.load(open(file_path, 'r'), Loader=loader))
    main(args)
>>>>>>> main
