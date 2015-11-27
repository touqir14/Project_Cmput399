function net = cnntrain(net, x, y, opts)
    m = size(x, 3); % size of training samples
    numbatches = m / opts.batchsize;
    if rem(numbatches, 1) ~= 0
        error('numbatches not integer');
    end
	
    net.rL = [];
    for i = 1 : opts.numepochs
        disp('Training start');
        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)]);
        tic;
        kk = randperm(m); %randomize the order of samples
        for l = 1 : numbatches
            
			% get (batchsize) samples and their labels, from a randomized
			% order
            batch_x = x(:, :, kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            batch_y = y(:,    kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            
            net = cnnff(net, batch_x); % Feedforward by the input to compute the score
			
            net = cnnbp(net, batch_y); % Backpropagation with the labels to get the deritives
			
            net = cnnapplygrads(net, opts);% update w's and b's
            if isempty(net.rL)
                net.rL(1) = net.L; % Loss
            end
            net.rL(end + 1) = 0.99 * net.rL(end) + 0.01 * net.L; % Store the loss, for drawing graphs 
        end
        toc;%display elapsed time
    end
    
end

