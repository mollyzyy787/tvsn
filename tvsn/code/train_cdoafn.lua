require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'optim'
require 'image'
require 'paths'
require 'nngraph'
require 'stn'
w_init = require 'weight-init'

opt = lapp[[
  --dataset						(default 'doafn')
	--split							(default 'train')
  --saveFreq          (default 20)
  --modelString       (default 'CDOAFN_SYM')
  -g, --gpu           (default 0)
  --imgscale          (default 256)
  --background				(default 0)
  --nThreads					(default 8)
  --maxEpoch          (default 200)
  --iter_per_epoch		(default 10)
  --batchSize         (default 25)
  --lr								(default 0.0001)
  --beta1							(default 0.9)
  --weightDecay       (default 0.0005)
	--data_dir					(default '../data/')
  --category          (default 'car')
	--resume						(default 0)
	-d, --debug					(default 0)
]]

print(opt)
if opt.debug > 0 then
	debugger = require('fb.debugger')
	debugger.enter()
end

-- initial setup
opt.manualSeed = torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setdefaulttensortype('torch.FloatTensor')

opt.modelName = string.format('%s_%s_%s_bs%03d', opt.modelString, opt.imgscale, opt.category, opt.batchSize)
opt.modelPath = '../snapshots/' .. opt.modelName
if not paths.dirp(opt.modelPath) then
  paths.mkdir(opt.modelPath)
end
if not paths.dirp(opt.modelPath .. '/training/') then
  paths.mkdir(opt.modelPath .. '/training/')
end

if opt.gpu < 0 or opt.gpu > 3 then opt.gpu = -1 end
if opt.gpu >= 0 then
  cutorch.setDevice(opt.gpu+1)
  print('<gpu> using device ' .. opt.gpu)
end

opt.metadata = string.format('../data/metadata_%s.cache',opt.category)

-- create data loader
print('loading maps...')
local maps = torch.load(opt.data_dir .. string.format('../data/maps_%s.t7',opt.category) )
opt.maps = maps.maps
opt.map_indices = maps.map_indices
print(string.format('Initializing data loader nthread:%d',opt.nThreads))
local DataLoader = paths.dofile('data/data.lua')
local data = DataLoader.new(opt.nThreads, opt.dataset, opt)
local data_sizes = data:size()
local ntrain = data_sizes[2]
local ntest = data_sizes[3]
print("Dataset: " .. opt.dataset .. " nTotal: " .. ntrain+ntest .. " nTrain: " .. ntrain .. " nTest: " .. ntest)

optimState = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
print(optimState)

-- load model from current learning stage
local epoch = 0
if opt.resume > 0 then
	for i = opt.maxEpoch, 1, -opt.saveFreq do
		if paths.filep(opt.modelPath .. string.format('/net-epoch-%d.t7', i)) then
			epoch = i
			local loader = torch.load(opt.modelPath .. string.format('/net-epoch-%d.t7', i))
			net = loader.net
			loss_list_im = loader.loss_list_im
			loss_list_map = loader.loss_list_map
			loss_list_map2 = loader.loss_list_map2
			print(opt.modelPath .. string.format('/net-epoch-%d.t7', i))
			print(string.format('resuming from epoch %d', i))
			break
		end
	end
else
	-- create model
	print('creating model...')
	local net_module = dofile('models/' .. string.format('%s_%s.lua', opt.modelString, opt.imgscale))
	net = net_module.create(opt)
	w_init.nngraph(net, 'kaiming')
	loss_list_im = torch.Tensor(1,1):fill(100)
	loss_list_map = torch.Tensor(1,1):fill(100)
	loss_list_map2 = torch.Tensor(1,1):fill(100)
end
local criterion_im = nn.AbsCriterion()
local criterion_map = nn.BCECriterion()
local criterion_map2 = nn.BCECriterion()
if opt.gpu >= 0 then
  net = net:cuda()
	criterion_im = criterion_im:cuda()
	criterion_map = criterion_map:cuda()
	criterion_map2 = criterion_map2:cuda()
end
local parameters, gradParameters = net:getParameters()


-- create closure to evaluate f(X) and df/dX of discriminator (Xiaobai: not discriminator, but doafn?)
local opfunc = function(x)
  	collectgarbage()
	gradParameters:zero()

	f = net:forward({batch_im_in, batch_view_in})

	im_err = criterion_im:forward(f[1], batch_im_out)
	local df_d_im = criterion_im:backward(f[1], batch_im_out)

	map_err = criterion_map:forward(f[2], batch_map)
	local df_d_map = criterion_map:backward(f[2], batch_map)

	map2_err = criterion_map2:forward(f[3], batch_map2)
	local df_d_map2 = criterion_map2:backward(f[3], batch_map2)

	net:backward({batch_im_in, batch_view_in}, {df_d_im:mul(2),df_d_map,df_d_map2})

	return 0, gradParameters
end

-- timer
local tm = torch.Timer()
local data_tm = torch.Timer()

for t = epoch+1, opt.maxEpoch do
	-- training
  net:training()
	for i = 1, opt.iter_per_epoch do
		local iter = i+(t-1)*opt.iter_per_epoch
		data_tm:reset(); data_tm:resume()
		batch_im_in, batch_im_out, batch_map, batch_view_in = data:getBatch()
		batch_map2 = torch.sum(batch_im_out,2) -- calculate contour from batch_im_out
		batch_map2 = torch.floor(batch_map2:mul(1/3))
		batch_map2 = batch_map2:add(-1):mul(-1)
		-- make it [0, 1] -> [-1, 1]
		batch_im_in:mul(2):add(-1)
    	batch_im_out:mul(2):add(-1)

		if opt.gpu >= 0 then
			batch_im_in = batch_im_in:cuda()
			batch_im_out = batch_im_out:cuda()
			batch_map = batch_map:cuda()
			batch_map2 = batch_map2:cuda()
			batch_view_in = batch_view_in:cuda()
		end
		data_tm:stop()

		tm:reset(); tm:resume()
		optim.adam(opfunc, parameters, optimState)
		tm:stop()
		print(string.format('#### epoch (%d)\t iter (%d) \t TrainError(IM/Mask)=(%.4f,%.4f,%.4f)'
							.. ' Time: %.3f DataTime: %.3f ', t, iter, im_err, map_err, map2_err,
							tm:time().real, data_tm:time().real))
		loss_list_im = torch.cat(loss_list_im, torch.Tensor(1,1):fill(im_err),1)
		loss_list_map = torch.cat(loss_list_map, torch.Tensor(1,1):fill(map_err),1)
		loss_list_map2 = torch.cat(loss_list_map2, torch.Tensor(1,1):fill(map2_err),1)
		-- plot 
		if iter % 1 == 0 then
			to_plot={}
			for k=1,opt.batchSize do
				to_plot[(k-1)*7 + 1] = f[1][k]
				to_plot[(k-1)*7 + 1]:add(1):mul(0.5) --generated image
				to_plot[(k-1)*7 + 2] = f[2][k]:repeatTensor(3,1,1) --generated vis map
				to_plot[(k-1)*7 + 3] = f[3][k]:repeatTensor(3,1,1) --generated contour map
				to_plot[(k-1)*7 + 4] = batch_im_in[k] --source image
				to_plot[(k-1)*7 + 4]:add(1):mul(0.5)
				to_plot[(k-1)*7 + 5] = batch_im_out[k] --target image
				to_plot[(k-1)*7 + 5]:add(1):mul(0.5)
				to_plot[(k-1)*7 + 6] = batch_map[k]:repeatTensor(3,1,1) --true vis map
				to_plot[(k-1)*7 + 7] = batch_map2[k]:repeatTensor(3,1,1) --true contour map
			end
			formatted = image.toDisplayTensor({input=to_plot, nrow = 7})
			image.save((opt.modelPath .. '/training/' .. 
					string.format('training_output_%05d.jpg',iter)), formatted)
			--io.stdin:read('*l')
		end
	end
	
  if t % opt.saveFreq == 0 then
			collectgarbage()
			parameters, gradParameters = nil, nil
			torch.save(opt.modelPath .. string.format('/net-epoch-%d.t7', t),
				{net = net:clearState(), loss_list_im = loss_list_im, loss_list_map = loss_list_map, loss_list_map2 = loss_list_map2})
			parameters, gradParameters = net:getParameters()
  end
end
