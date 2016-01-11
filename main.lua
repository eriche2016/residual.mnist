require 'nn'
--[[
-- if you want to load a pretrained model, then the paramters must match, finetune on this model 
-- currenly, this is my parameter settings, which support 36 layers as i have pretrained a 36 layers model 
-- but you can train your model as many layers you like from scratch 
-- a model of 36 layers, also note that now currenlty only support gpu training
--]]

local train = require 'train'
local model = require 'model'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('MNIST training using Residual Neural Networks')
cmd:text('Example:')
cmd:text('$> th main.lua -layers 100 -batchSize 128 -epochs 10')
cmd:text('Options:')
cmd:option('-seed', 1234, 'seed of the rng')
cmd:option('-momentum', 0.9, 'momemtum during SGD')
cmd:option('-learningRate', 0.1, 'learning rate at t=0')
cmd:option('-learningRateDecay', 5.0e-6, 'learning rate decay')
cmd:option('-epochs', 1, 'number of epochs to run')
cmd:option('-batchSize', 64, 'batch size(adjust to fit in GPU)')
cmd:option('-layers', 36, 'approx num of layers to train')
cmd:option('-silent', false, 'whether need to print the parameters')
cmd:option('-init_from', '', 'specify the pretrained model to do initialization')
cmd:option('-gpuid', 1, 'specify the gpuid, -1 means cpu training')
cmd:text()

opt = cmd:parse(arg or {})

if not opt.silent then
   print(opt)
end 

torch.manualSeed(opt.seed)
-- currently no gpu
torch.setdefaulttensortype('torch.FloatTensor')
if opt.gpuid >= 0 then 
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then print('cunn not found') end 
    if not ok2 then print('cutorch not found') end 
    if ok and ok2 then 
        print('using cuda on gpu ' .. opt.gpuid .. '...')
        cutorch.setDevice(opt.gpuid + 1) -- torch is 1-indexed
        cutorch.manualSeed(opt.seed)
    else 
        print('cannnot run in the gpu mode because of cuda is not installed correctly or cunn and cutorch are not installed')
        print('falling back to cpu mode')
        opt.gpuid = -1 
    end 
end


local N = (opt.layers-10)/6  -- N = (36-10)/6=26/6=4.33333..

-- prepare data
local mnist = require 'mnist'
local train_set = mnist.traindataset()

 Xt = train_set.data
 Yt = train_set.label

local test_set = mnist.testdataset()
Xv = test_set.data
Yv = test_set.label

Yt[Yt:eq(0)] = 10
Yv[Yv:eq(0)] = 10

if string.len(opt.init_from) == 0 then 
    net,ct = model.residual(N)
else 
    checkpoint =  torch.load(opt.init_from)
    net = checkpoint.protos.net
    ct = checkpoint.protos.ct 
end

if opt.gpuid >= 0 then 
   -- actually if loaded from pretrained model
   -- then the net and ct are aready in the gpu, because they are stored in cuda datatype
   -- shift model to gpu
   net = net:cuda()     
   -- is need to shift criterion to gpu
   ct = ct:cuda() 
end 

print(net:__tostring__())

local sgd_config = {
      learningRate = opt.learningRate,
      learningRateDecay = opt.learningRateDecay,
      momentum = opt.momemtum
   }

print('Number of convolutional layers .. '..#net:findModules('nn.SpatialConvolution'))
train.sgd(net,ct,Xt,Yt,Xv,Yv,opt.epochs,sgd_config,opt.batchSize)

