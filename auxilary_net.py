'''

使用vgg构建一个网络
在网络的上面定义一层loss 
定义一个trainer 

如果train 就 
     def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If your want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        """
        If your want to do something with the losses, you can wrap the model.
        """
        loss_dict = self.model(data)
        losses = sum(loss for loss in loss_dict.values())
        self._detect_anomaly(losses, loss_dict)

        metrics_dict = loss_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        """
        If you need accumulate gradients or something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()
        losses.backward()

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method.
        """
        self.optimizer.step()
 
 如果eval 
    使用model,eval / 或者不apply loss， 只输出weighted score


总结： model.train() 和 model.eval() 一般在模型训练和评价的时候会加上这两句，主要是针对由于model 在训练时和评价时 Batch Normalization 和 Dropout 方法模式不同；因此，在使用PyTorch进行训练和测试时一定注意要把实例化的model指定train/eval；



'''