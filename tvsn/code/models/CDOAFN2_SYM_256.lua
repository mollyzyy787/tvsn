require 'nngraph'
require 'stn'

local GFN = {}

function GFN.create(opts)
	local inputs = {}
	local input_im = nn.Identity()()
	local input_view = nn.Identity()()
	table.insert(inputs,input_im)
	table.insert(inputs,input_view)

	-- encoder
  -- 256 x 256 x 3 --> 128 x 128 x 16
	local en_conv1 = nn.ReLU()(nn.SpatialConvolution(3,16,5,5,2,2,2,2)(input_im))
  -- 128 x 128 x 16 --> 64 x 64 x 32
	local en_conv2 = nn.ReLU()(nn.SpatialConvolution(16,32,5,5,2,2,2,2)(en_conv1))
  -- 64 x 64 x 32 --> 32 x 32 x 64
	local en_conv3 = nn.ReLU()((nn.SpatialConvolution(32,64,5,5,2,2,2,2)(en_conv2)))
  -- 32 x 32 x 64 --> 16 x 16 x 128
	local en_conv4 = nn.ReLU()((nn.SpatialConvolution(64,128,3,3,2,2,1,1)(en_conv3)))
  -- 16 x 16 x 128 --> 8 x 8 x 256
	local en_conv5 = nn.ReLU()((nn.SpatialConvolution(128,256,3,3,2,2,1,1)(en_conv4)))
  -- 8 x 8 x 256 --> 4 x 4 x 512
	local en_conv6 = nn.ReLU()((nn.SpatialConvolution(256,512,3,3,2,2,1,1)(en_conv5)))
  -- 4 x 4 x 512 --> 2048
	local en_fc6 = nn.ReLU()(nn.Linear(4*4*512,2048)(nn.Reshape(4*4*512)(en_conv6)))

	-- view 
	local view_fc1 = nn.ReLU()(nn.Linear(17,128)(input_view))
	local view_fc2 = nn.ReLU()(nn.Linear(128,256)(view_fc1))
	local view_concat = nn.JoinTable(2)({en_fc6,view_fc2})

	-- decoder 
	local de_fc1 = nn.ReLU()(nn.Linear(2048+256,2048)(view_concat))
	if opts.dropout == 1 then
		de_fc1 = nn.Dropout(0.5)(de_fc1)
	end
	local de_fc2 = nn.ReLU()(nn.Linear(2048,2048)(de_fc1))
	if opts.dropout == 1 then
		de_fc2 = nn.Dropout(0.5)(de_fc2)
	end
	local de_fc3 = nn.Reshape(512,4,4)(nn.ReLU()(nn.Linear(2048,4*4*512)(de_fc2)))

	-- 4 x 4 x 512 --> 8 x 8 x 256
	local de_deconv1 = nn.ReLU()(nn.SpatialFullConvolution(512,256,3,3,1,1,1,1)(nn.SpatialUpSamplingNearest(2)(de_fc3)))
	-- 8 x 8 x 256 --> 16 x 16 x 128
	local de_deconv2 = nn.ReLU()(nn.SpatialFullConvolution(256,128,3,3,1,1,1,1)(nn.SpatialUpSamplingNearest(2)(de_deconv1)))
	-- 16 x 16 x 128 --> 32 x 32 x 64
	local de_deconv3 = nn.ReLU()(nn.SpatialFullConvolution(128,64,3,3,1,1,1,1)(nn.SpatialUpSamplingNearest(2)(de_deconv2)))
	-- 32 x 32 x 64 --> 64 x 64 x 32 
	local de_deconv4 = nn.ReLU()(nn.SpatialFullConvolution(64,32,5,5,1,1,2,2)(nn.SpatialUpSamplingNearest(2)(de_deconv3)))
	-- 64 x 64 x 32--> 128 x 128 x 16
	local de_deconv5 = nn.ReLU()(nn.SpatialFullConvolution(32,16,5,5,1,1,2,2)(nn.SpatialUpSamplingNearest(2)(de_deconv4)))
	-- 128 x 128 x 16--> 128 x 128 x 3
	local de_deconv6 = nn.Tanh()(nn.SpatialFullConvolution(16,2,5,5,1,1,2,2)(nn.SpatialUpSamplingNearest(2)(de_deconv5)))
	local flow = nn.Transpose({3,4})(nn.Transpose({2,3})(de_deconv6))
	local im = nn.Transpose({3,4})(nn.Transpose({2,3})(input_im))
	local sample = nn.BilinearSamplerBHWD()({im,flow})
	local sample = nn.Transpose({3,2})(nn.Transpose({4,3})(sample))

	local mask = nn.Sigmoid()(nn.SpatialFullConvolution(16,1,5,5,1,1,2,2)(nn.SpatialUpSamplingNearest(2)(de_deconv5)))
	local mask2 = nn.Sigmoid()(nn.SpatialFullConvolution(16,1,5,5,1,1,2,2)(nn.SpatialUpSamplingNearest(2)(de_deconv5)))

	local outputs = {}
	table.insert(outputs,sample)
	table.insert(outputs,mask)
	table.insert(outputs,mask2)

	return nn.gModule(inputs, outputs)
end

return GFN 
