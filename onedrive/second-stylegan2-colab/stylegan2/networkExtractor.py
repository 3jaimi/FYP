import pickle
import dnnlib
import dnnlib.tflib as tflib

def load_stylegan2_networks(pkl_path):
    # Load network pickle file
    with open(pkl_path, 'rb') as f:
        _, _, Gs = pickle.load(f)

    # Extract individual networks from Gs
    G = Gs.components.synthesis
    D = Gs.components.discriminator

    return G, Gs, D

if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract networks from StyleGAN2 .pkl file')
    parser.add_argument('pkl_path', type=str, help='Path to the .pkl file containing StyleGAN2 networks')
    args = parser.parse_args()

    # Load networks from the specified .pkl file
    G, Gs, D = load_stylegan2_networks(args.pkl_path)

    # Print some information about the loaded networks
    print("Generator network (G):", G)
    print("Generator with mapping network (Gs):", Gs)
    print("Discriminator network (D):", D)