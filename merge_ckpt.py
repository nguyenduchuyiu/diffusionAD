import torch
import argparse

def merge_checkpoints(params_path, seg_path, output_path=None):
    params_ckpt = torch.load(params_path, map_location='cpu')
    seg_ckpt = torch.load(seg_path, map_location='cpu')
    
    params_ckpt['seg_model_state_dict'] = seg_ckpt['seg_model_state_dict']
    
    if output_path is None:
        output_path = params_path
    torch.save(params_ckpt, output_path)
    print(f"Merged checkpoint saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', required=True, help='Path to params-last.pt')
    parser.add_argument('--seg', required=True, help='Path to seg-last.pt')
    parser.add_argument('--output', default=None, help='Output path (default: overwrite params)')
    args = parser.parse_args()
    
    merge_checkpoints(args.params, args.seg, args.output)

