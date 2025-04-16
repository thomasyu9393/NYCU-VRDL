import os
import argparse
from model import Model
from dataset import get_dataset, get_dataloader


def main(args):
    # Remember to change the model name!!!
    model_name = 'fasterrcnn_resnet50_fpn_v2'
    model = Model(model_name=model_name)

    if args.train:
        train_dataset = get_dataset(args, mode='train')
        valid_dataset = get_dataset(args, mode='valid')
        train_loader = get_dataloader(args, train_dataset, mode='train')
        valid_loader = get_dataloader(args, valid_dataset, mode='valid')

        if args.ckpt is not None:
            model.load_weights(args.ckpt)

        model.train(
            train_loader, valid_loader,
            num_epochs=args.epochs, lr=args.lr, ckpt_dir=args.ckpt_dir)

    if args.test:
        test_dataset = get_dataset(args, mode='test')
        test_loader = get_dataloader(args, test_dataset, mode='test')

        if args.train:
            model.load_weights(os.path.join(args.ckpt_dir, "best_model.pth"))
        elif args.ckpt is not None:
            model.load_weights(args.ckpt)
        else:
            model.load_weights(os.path.join(args.ckpt_dir, "last_model.pth"))

        model.test(test_loader, zip_out=args.zip_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--test_ensemble', action='store_true')
    parser.add_argument('--data_dir', type=str, default='nycu-hw2-data')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--ckpt_dir', type=str, default='ckpt')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--zip_file', type=str, default='submission.zip')
    args = parser.parse_args()

    # Set GPU ID
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    main(args)
    print("main: End of main")
