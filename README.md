# Signal Aware Data Parallel Training

This is a repo I made to build some understanding of what happens under the hood of the DDP module in Pytorch.

A basic premise distributed data parallel is that each worker in your setup gets data of roughly equal quality.

However, of the time in modern LLM training datasets that are scrapes of the internet, there is an incredible amount of repeated infomration, such that this assumption is not accurate.

I tried to investigate what would happen if we trained a model while keeping some measure of the data quality (signal) on each worker.

My initial experiments probably don't have enough nodes (I'm only using 2 GPUs as of now because I'm broke), so downweighting one node tends to lose a lot of useful gradients.

I'll probably try scaling this experiment up with 8 GPUs if I ever get the resources.

 
