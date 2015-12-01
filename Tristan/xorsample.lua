--[[
Sample Multi-Layer Perceptron (Neural Network) in Torch7
Author: Tristan Meleshko

References:
	torch.ch/docs/
	torch5.sorceforge.net/manual/newbieTutorial.html
--]]

-- Run using 'th mnist_cnn.lua'
require 'cunn'
require 'torch'

--[[*** Configuration Section ***]]--

-- General settings
use_cuda = 1			-- Disable this to turn off CUDA GPU support

-- Function parameters
ninput = 2				-- Number of inputs the function accepts
noutput = 2				-- Number of outputs the function produces

-- Trainer info
learningRate = 0.01		-- Amount to update by in stochastic gradient descent
maxIterations = 25		-- Maximum number of passes over the dataset [25]
shuffleIndices = false  -- Randomize order of training data? [nil]

-- Training info
epochs = 4			-- Alternate between training and testing this many times
ntraining = 1000	-- Number of training inputs
ntesting = 100		-- Number of validation set inputs (unused in this sample)

--[End of general configurations]--
if use_cuda then require 'cutorch' end
mlp = nn.Sequential()

--[[Add Layers]]--
mlp:add(nn.Linear(ninput, 100))
mlp:add(nn.Tanh())

mlp:add(nn.Linear(100, 100))
mlp:add(nn.Tanh())

mlp:add(nn.Linear(100, 100))
mlp:add(nn.Tanh())

mlp:add(nn.Linear(100, noutput))

--[[*** End of Configuration ***]]--

if use_cuda then
	mlp:cuda()
	-- Add layers to automatically convert results
	mlp_auto = nn.Sequential()
	-- Copies input float tensors and sends them to CUDA
	mlp_auto:add(nn.Copy('torch.DoubleTensor', 'torch.CudaTensor'))
	-- Add our neural network
	mlp_auto:add(mlp)
	-- Copy back from CUDA tensors into float tensors for our results
	mlp_auto:add(nn.Copy('torch.CudaTensor', 'torch.DoubleTensor'))
	-- Use this as our MLP
	mlp = mlp_auto
end

--[[Construct Dataset]]--
dataset={};
-- dataset must expose a :size() method
function dataset:size() return ntraining; end

-- Dataset to learn will be the XOR function
for i = 1,dataset:size() do
	local input = torch.randn(2);
	local output = torch.Tensor(1);
	-- Learning the XOR function
	if input[1]*input[2] > 0 then
		-- output is low if both inputs are high
		output = torch.Tensor({0, 1})
	else
		-- output is high if both inputs are low
		output = torch.Tensor({1, 0})
	end
	-- Append to the training set
	dataset[i] = {input, output};
end


--[[Simple Training Section]]--
-- Minimize the function with respect to the mean square error
criterion = nn.MSECriterion()

-- train using Stochastic Gradient Descent
trainer = nn.StochasticGradient(mlp, criterion)
trainer.learningRate = learningRate
trainer.maxIterations = maxIterations

-- Train the dataset
trainer:train(dataset)

--[[Manual Training Section]]--
--Same as the section before, but all the work is done manually
criterion = nn.MSECriterion()

for e = 1,epochs do
for t = 1,maxIterations do
	totalError = 0

	for i = 1,ntraining do
		input = dataset[i][1]
		output = dataset[i][2]

		-- Feed forward into the network
		prediction = mlp:forward(input)
		-- Get the error amount
		err = criterion:forward(prediction, output)
		totalError = totalError + err;

		-- Zero the gradient accumulator
		mlp:zeroGradParameters()
		-- Accumulate Gradients
		gradient = criterion:backward(prediction, output)
		-- Backpropogate the errors
		mlp:backward(input, gradient)
		-- Update the network using the learning rate
		mlp:updateParameters(learningRate)
	end
	print('current error =', totalError / ntraining)
end
end

--[[Output Results]]--
function realOut(x)
	return mlp:forward(torch.Tensor(x))
end
function predict(x)
	pred = realOut(x)
	if pred[1] > pred[2] then return 1
	else return 0
	end
end

print('\n\n===== Learning Results =====')
print('[0 0] ->\n', realOut({0,0}), '(expected 0,1)')
print('[1 0] ->\n', realOut({1,0}), '(expected 1,0)')
print('[0 1] ->\n', realOut({0,1}), '(expected 1,0)')
print('[1 1] ->\n', realOut({1,1}), '(expected 0,1)')

print('\n\n===== Output Values =====')
print('[0 0] ->', predict({0,0}), '(expected 0)')
print('[1 0] ->', predict({1,0}), '(expected 1)')
print('[0 1] ->', predict({0,1}), '(expected 1)')
print('[1 1] ->', predict({1,1}), '(expected 0)')
