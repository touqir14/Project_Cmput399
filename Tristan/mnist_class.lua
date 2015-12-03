--[[
Convolutional Network for the MNIST dataset
Author: Tristan Meleshko

Run using 'th mnist_cnn.lua'. If you change the model,
you must delete the data folder.
--]]

require 'cunn'
require 'torch'
require 'image'

-- Command-line options
local opt = lapp[[
	-t,--threads	(default 8)		number of threads
	-s,--seed	(default 1)		random seed
	-g,--gpu	use CUDA
	-r,--reset	reset the training from scratch (don't load)
	-p,--permute	Use random permutations of the training data
]]
torch.manualSeed(opt.seed)
torch.setnumthreads(opt.threads)
use_cuda = opt.gpu

print('[Using ' .. torch.getnumthreads() .. ' threads...]')
print('[Using ' .. torch.getnumcores() .. ' cores...]')
if use_cuda then print('[CUDA enabled]') end

print('\n')

--[[*** Configuration Section ***]]--
-- Function parameters
classes = {'0','1','2','3','4','5','6','7','8','9'}	-- Target classes 
geometry = {32, 32}	-- size of the images

-- Trainer info
learningRate = 0.01		-- Amount to update by in stochastic gradient descent
shuffleIndices = false  -- Randomize order of training data? [nil]

-- Training info
epochs = 100		-- Alternate between training and testing this many times
batchSize = 10		-- Number of items in a training/testing batch

ntraining = 60000	-- Number of training inputs
ntesting = 10000	-- Number of validation set inputs (unused in this sample)

--[End of general configurations]--
model = nn.Sequential()

--[[Add Layers]]--
-- Ref: github.com/namin/torch-demos/

--Remark: SpatialSutractiveNormalization doesn't work well with CUDA.
-- 		  nor does SpatialConvolutionMap. Neither expose an update
--		  function for CUDA.

--model:add(nn.SpatialSubtractiveNormalization(1, image.gaussian1D(15)))
model:add(nn.SpatialConvolutionMM(1,32, 5,5))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2,2,2,2))

--model:add(nn.SpatialSubtractiveNormalization(32, image.gaussian1D(15)))
model:add(nn.SpatialConvolutionMM(32,128, 5,5))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2,2,2,2))

model:add(nn.Reshape(128*5*5))
model:add(nn.Dropout())
model:add(nn.Linear(128*5*5, 256))
model:add(nn.Tanh())
model:add(nn.Dropout())
model:add(nn.Linear(256, 256))
model:add(nn.Tanh())
model:add(nn.Dropout())
model:add(nn.Linear(256, #classes))
model:add(nn.SoftMax())

--[[*** End of Configuration ***]]--

if use_cuda then
	require 'cutorch'
	-- Put our model onto the GPU
	model = model:cuda()
	-- Add layers to automatically convert results
	model_auto = nn.Sequential()
	-- Copies input float tensors and sends them to CUDA
	model_auto:add(nn.Copy('torch.DoubleTensor', 'torch.CudaTensor'))
	-- Add our neural network
	model_auto:add(model)
	-- Copy back from CUDA tensors into float tensors for our results
	model_auto:add(nn.Copy('torch.CudaTensor', 'torch.DoubleTensor'))
	-- Use this as our model
	model = model_auto
end

--[[Load Dataset]]--
--Reference: github.com/torch/tutorials/blob/master/A_datasets/mnist.lua
tar = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/mnist.t7.tgz'
if not paths.dirp('mnist.t7') then
	-- works on UNIX systems
	os.execute('wget ' .. tar)
	os.execute('tar xvf ' .. paths.basename(tar))
end

train_file = 'mnist.t7/train_32x32.t7'
test_file = 'mnist.t7/test_32x32.t7'

train_data = torch.load(train_file, 'ascii')
test_data = torch.load(test_file, 'ascii')

--[[Workers]]--

-- Minimize the function with respect to the mean square error
criterion = nn.MSECriterion()

function runValidation(dataset)
	local totalError = 0
	local batchNum = 1
	local accuracy = 0
	local confusion = torch.Tensor(#classes, #classes)
	confusion[{}] = 0

	-- Generate a random perm of the training data
	if opt.permute then dataset = setupDataset(test_data, ntraining) end

	print('Validation Epoch ' .. epoch)
	for t = 1,ntesting,batchSize do
		bsize = math.min(t+batchSize, dataset:size()) - t
		-- Need to hold targets and outputs for back prop
		local inputs = torch.Tensor(bsize, 1, geometry[1],geometry[2])
		local targets = torch.Tensor(bsize, #classes)
		local labels = torch.Tensor(bsize)
		-- For each element in the batch do
		local k = 1
		for i = t,t+bsize-1 do
			-- Add the inputs to the batch
			local input = dataset[i][1]
			local output = dataset[i][2]
			inputs[k] = input
			targets[k] = output
			_,labels[k] = torch.max(output,1)
			k = k + 1
		end

		-- Get the error in the prediction
		model:evaluate()
		local outputs = model:forward(inputs)
		local predictions
		_,predictions = torch.max(outputs,2)

		predictions = predictions:int()
		labels = labels:int()

		local batchError = criterion:forward(targets, outputs)
		totalError = totalError + (batchError - totalError)/batchNum

		-- Determine the actual accuracy of the model
		matches = predictions:eq(labels)
		accuracy = accuracy + torch.sum(matches)
		-- Count where misclassification happens
		for a = 1,predictions:numel() do
			i = labels[a]
			j = predictions[a][1]
			confusion[i][j] = confusion[i][j]+1
		end

		-- Output the error amount
		print('| validation batch %d.%d'%{epoch,batchNum}
			..' error = %.4f%%' % (batchError*100))
		batchNum = batchNum + 1
	end
	print('Validation error = %.4f%%' % (totalError*100))
	print('Accuracy = %.4f%%' % (accuracy/ntesting*100))
	print('\nConfusion Matrix:')
	print('     ', table.concat(classes, '     '))
	print(string.rep('-',70))
	print(confusion)
end

function saveModel()
	if not paths.dirp('data') then os.execute('mkdir data'); end
	torch.save('data/model-epoch' .. epoch .. '.dat', model)
	torch.save('data/model.dat', model)

	info = {}
	info['epoch'] = epoch
	torch.save('data/info.dat', info)
	print('saved model info')
end

function loadModel()
	if not opt.reset and paths.filep('data/model.dat') then
		model = torch.load('data/model.dat')
		info = torch.load('data/info.dat')

		startEpoch = info.epoch + 1
		print('Loaded model from file')
	else
		startEpoch = 1
		print('Starting new training set')
	end
end

function postEpoch()
	--[[
	After an epoch, run validation and save the state of the
	network so that we can stop it and restart it later.
	--]]
	runValidation(testData)
	saveModel()
end

-- Converts the mnist data to the expected form
function setupDataset(inputData, n)
	dataset = {}
	function dataset:size() return n; end
	perm = torch.randperm((#inputData['data'])[1])
	for j = 1,n do
		i = perm[j]
		-- input is an image
		input = inputData['data'][i]:double()

		-- output is 1 if at corresponding classification
		output = torch.Tensor(#classes)
		output[{}] = 0
		output[inputData['labels'][i]] = 1
		dataset[j] = {input, output}
	end
	return dataset
end

--[[Setup Dataset]]--
trainData = setupDataset(train_data, ntraining)
testData = setupDataset(test_data, ntesting)

--[[Training Section]]--
loadModel()
criterion = nn.MSECriterion()

function train(dataset)
print('=-=-=-=-=-=-=-=-=')
print('|Training model:|')
print('=---------------=')
print(model)
print('=-=-=-=-=-=-=-=\n')
local totalTime = sys.clock()
for e = startEpoch,epochs do
	epoch = e
	local time = sys.clock()
	local totalError = 0
	local batchNum = 1	

	-- Generate a random perm of the training data
	if opt.permute then dataset = setupDataset(train_data, ntraining) end

	print('Training Epoch '..e)
	-- For each batch do
	for t = 1,dataset:size(),batchSize do
		bsize = math.min(t+batchSize, dataset:size()) - t
		-- Need to hold targets and outputs for back prop
		local inputs = torch.Tensor(bsize, 1, geometry[1],geometry[2])
		local targets = torch.Tensor(bsize, #classes)
		-- For each element in the batch do
		local k = 1
		for i = t,t+bsize-1 do
			-- Add the inputs to the batch
			local input = dataset[i][1]
			local output = dataset[i][2]
			inputs[k] = input
			targets[k] = output
			k = k + 1
		end
		collectgarbage()
		
		-- Compute the forward error
		model:training()
		local outputs = model:forward(inputs)

		-- Zero the gradient accumulator
		model:zeroGradParameters()
		-- Accumulate Gradients
		local gradient = criterion:backward(outputs, targets)
		-- Backpropogate the errors
		model:backward(inputs, gradient)
		-- Update the network using the learning rate
		model:updateParameters(learningRate)
	
		-- Update batch info
		local batchError = criterion:forward(outputs, targets)
		--totalError = average error over all batches
		totalError = totalError + (batchError-totalError)/batchNum
		print('| training batch %d.%d' % {epoch,batchNum} .. 
			' error = %.4f%%' % (batchError*100))
		batchNum = batchNum+1
	end
	print('Training error = %.4f%%' % (totalError*100), '\n')
	postEpoch()

	elapsedTime = sys.clock() - time
	print('=== End of Epoch '..e..' --- %2.2f' % elapsedTime..' s ===\n\n')
end
print("\nTotal Time: %.2f" % (sys.clock() - totalTime))
end

collectgarbage()
train(trainData)

--[[Output Results]]--
