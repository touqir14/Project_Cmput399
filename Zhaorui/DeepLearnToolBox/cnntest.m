function [net,er, bad, h, a] = cnntest(net, x, y)
    net = cnnff(net, x); % 前向传播得到输出
	% [Y,I] = max(X) returns the indices of the maximum values in vector I
    [~, h] = max(net.o); % 找到最大的输出对应的标签
    [~, a] = max(y); 	 % 找到最大的期望输出对应的索引
    bad = find(h ~= a);  % find the number of times of misclassifying
    er = numel(bad) / size(y, 2); % 计算错误率
end
