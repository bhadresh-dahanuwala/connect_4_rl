import argparse
from connect_4_rl.trainer import Trainer

def main():
    parser = argparse.ArgumentParser(description="Train Connect-4 AlphaZero Agent")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of iterations to run")
    parser.add_argument("--self-play-games", type=int, default=100, help="Number of games to play in self-play")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for the training phase")
    parser.add_argument("--eval-games", type=int, default=20, help="Number of games to be played in the evaluation phase")
    parser.add_argument("--simulations", type=int, default=50, help="MCTS simulations per move")
    parser.add_argument("--cpuct", type=float, default=2.0, help="MCTS cpuct parameter")
    parser.add_argument("--workers", type=int, default=10, help="Number of parallel workers for self-play and evaluation")
    parser.add_argument("--max-buffer-size", type=int, default=50000, help="Maximum number of examples to store in replay buffer")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size for training")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    args = parser.parse_args()

    config = {
        'iterations': args.iterations,
        'num_self_play_games': args.self_play_games,
        'epochs': args.epochs,
        'num_eval_games': args.eval_games,
        'num_simulations': args.simulations,
        'c_puct': args.cpuct,
        'workers': args.workers,
        'max_buffer_size': args.max_buffer_size,
        'batch_size': args.batch_size,
        'resume': args.resume
    }

    print("Training Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    trainer = Trainer(config)
    trainer.run()

if __name__ == "__main__":
    main()
