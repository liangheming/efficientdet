from processors.ddp_apex_processor import DDPApexProcessor

# python -m torch.distributed.launch --nproc_per_node=2 --master_port 50003 main.py

if __name__ == '__main__':
    processor = DDPApexProcessor(cfg_path="config/efficientdet.yaml")
    processor.run()
