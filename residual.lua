require 'nn'
require 'cunn'

-- note that net is of type nn.Sequential(), and will be passed as
-- a arguments, this function will add a residual part to the net 
local function convunit(net,fin,fout,fsize,str,pad,nobatch)
    local nobatch = nobatch or false
    local pad = pad or 1   -- padding size, by default is 1 
    local str = str or 1   -- stride, by default is 1
    local fsize = fsize or 3  -- kernel size, by default is 1 
    net:add(nn.SpatialConvolution(fin,fout,fsize,fsize,str,str,pad,pad))
    if(nobatch==false) then net:add(nn.SpatialBatchNormalization(fout)) end   -- batch normalization 
    net:add(nn.ReLU(true))  -- true=in-place, false=keeping separate state 
end


local function convunit31(net,fin,half,str,nobatch)
    local str = str or 3
    local half = half or false
<<<<<<< HEAD

    -- by default, half = false, if it is true, then 
    -- will half the feature map size(both along h and w), 
    -- but we will also need to double the number of feature maps 
    if(half) then                        
        convunit(net,fin,2*fin,str,2,nil,nobatch)
    else 
        convunit(net,fin,fin,str,1,nil,nobatch) end
=======
    if(half) then
        convunit(net,fin,2*fin,str,2,nil,nobatch)
    else convunit(net,fin,fin,str,1,nil,nobatch) end
>>>>>>> 71625ff8e6a86c8ba809f004250617616913d231
end

local function convunit2(net,fin,half)
    local half = half or false
    convunit31(net,fin,half,nil,true)
    if(half) then convunit31(net,2*fin) 
    else convunit31(net,fin) end  
end

local function resUnit(net, unit, fin, half)
    local half = half or false
    local net = net or nn.Sequential()
   
    local cat = nn.ConcatTable()
    cat:add(unit)
    if(half==false) then
        cat:add(nn.Identity())
    else 
        cat:add(nn.SpatialConvolution(fin,2*fin,1,1,2,2)) -- if half = true 
    end

    net:add(cat)  -- stack cat above net 
    net:add(nn.CAddTable())
    net:add(nn.ReLU(true))
    return net
end

--conv + residual  
local function rconvunit2(net,fin,half)
    local unit = nn.Sequential()
    convunit2(unit,fin,half)
    resUnit(net,unit,fin,half) -- stack unit on net 
end

local function rconvunitN(net,fin,N)
    local N = N or 0
    for i=1,N do
        rconvunit2(net,fin)  -- half = false 
    end
end

local res = {}
res.convunit = convunit
res.rconvunit2 = rconvunit2
res.rconvunitN = rconvunitN
return res
