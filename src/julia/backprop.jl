# backpropagation of fully connected dnn
# lixun 2017-11-07
#
#  X(4) => 8 => 8 => 4 fully connected nn
# a: activations, for example sigmoid(z) = sigmoid(aw+b) 
# w: weights
# b: biases
# L: nn depth
# z: wa+b

mutable struct net{T <: AbstractFloat}

    a::Array{Array{T,1},1}  # activations, 1st layer as input
    z::Array{Array{T,1},1}  # activation input, z[1] is not in use
    δ::Array{Array{T,1},1}  # errors, [1] is not in use

    w::Array{Array{T,2},1}  # weights, w[1] is not used
    b::Array{Array{T,1},1}  # biases, b[1] is not used

    ∂C∂w::Array{Array{T,2},1}  # ∂C/∂w of all weights, [1] is not in use
    ∂C∂b::Array{Array{T,1},1}  # ∂C/∂b of all biases, [1] is not in use

    L::Int64  # layers

    # for SGD
    ∇w::Array{Array{T,2},1}
    ∇b::Array{Array{T,1},1}
    
    function net{T}(layers) where T <: AbstractFloat
        L = length(layers)

        a = [zeros(T,i) for i in layers]
        z = deepcopy(a)
        δ = deepcopy(a)
    
        w = [[zeros(T,length(a[1]),1)]; [randn(T,length(a[i]),length(a[i-1])) for i = 2:L]]
        b = [[zeros(T,length(a[1]))]; [randn(T,length(a[i])) for i = 2:L]]
    
        ∂C∂w = deepcopy(w)
        ∂C∂b = deepcopy(b)
        ∇w = deepcopy(w)
        ∇b = deepcopy(b)

        new(a,z,δ,w,b,∂C∂w,∂C∂b,L,∇w,∇b)
    end
end



# x: Array{Array{T,1},1} for best performance
function backprop!(nn::net{T}, x::Array{T,1}, y::Array{T,1}) where T <: AbstractFloat

    L = nn.L

    # feedforward
    nn.a[1] = x
    for i = 2 : L
        nn.z[i] .= (nn.w[i] * nn.a[i-1]) .+ nn.b[i]
        nn.a[i] .= sigmoid.(nn.z[i])
    end
    # nn.a, nn.z

    # output error ∇aC ⦿ σ'(z[L])
    #nn.δ[L] .= ∂C∂a.(nn.a[L], y) .* sigmoid_prime.(nn.z[L])  # quadratic cost
    nn.δ[L] .= ∂C∂a.(nn.a[L], y)                              # cross entropy cost
    nn.∂C∂b[L] .= nn.δ[L]
    ∂C∂w!(nn.∂C∂w[L], nn.a[L-1], nn.δ[L])
   
    # for each l = L-1, L-2 ... 2 compute δ[l] = (transpose(w[l+1])δ[l+1]) ⦿ σ'(z[l])
    #    ∂C/∂w[l][j,k] = a[l-1][k] * δ[l][j]
    #    ∂C/∂b[l][j] = δ[l][j]
    for l = L-1:-1:2
        nn.δ[l] .= (transpose(nn.w[l+1]) * nn.δ[l+1]) .* sigmoid_prime.(nn.z[l])
        nn.∂C∂b[l] .= nn.δ[l]
        ∂C∂w!(nn.∂C∂w[l], nn.a[l-1], nn.δ[l])
    end
    nothing
end    


# η: learning rate, i.e. the scaling factor applied to the gradient ∂C/∂w and ∂C/∂b
# λ: regulatization factor for weight decaying
# n: total size of the training data set
function update_minibatch(nn::net{T}, minibatch::Array{Tuple{Array{T,1},Array{T,1}},1}, η::T, λ::T, n::Int64) where T <: AbstractFloat
    
    for i in nn.∇w
        fill!(i, zero(T))
    end
    for i in nn.∇b
        fill!(i, zero(T))
    end
    for (x,y) in minibatch
        backprop!(nn, x, y)
        for (i,j) in enumerate(nn.∇w)
            j .+= nn.∂C∂w[i]
        end
        for (i,j) in enumerate(nn.∇b)
            j .+= nn.∂C∂b[i]
        end
    end
    m = length(minibatch)
    α = one(T) - λ * η / n
    β = η / m
    nn.w .= α .* nn.w .- β .* nn.∇w
    nn.b .-= ((η/m) .* nn.∇b)
end


function feedforward(nn::net{T}, x::Array{T,1}) where T <: AbstractFloat
    nn.a[1] = x
    for i = 2 : nn.L
        nn.a[i] .= sigmoid.((nn.w[i] * nn.a[i-1]) .+ nn.b[i])
    end
    nn.a[nn.L]
end


# x=7 ∈ {0,1,...9} -> [0,0,0,0,0,0,0,1,0,0]
function onehot(T::DataType, n::Int64, x::Int64)
    v = zeros(T,n)
    v[x+1] = T(1)
    v
end

function accu(nn::net{T}, batch::Array{Tuple{Array{T,1},Int64},1}) where T <: AbstractFloat
    correct = 0
    for (i,j) in batch
        val, index = findmax(feedforward(nn, i))
        (index-1) == j && (correct += 1)
    end
    a = correct / length(batch)
    a
end

function totalcost(nn::net{T}, batch::Array{Tuple{Array{T,1},Array{T,1}},1}, λ::T) where T <: AbstractFloat
    
    n = length(batch)
    Σcost = zero(T)

    for (x,y) in batch
        a = feedforward(nn, x)
        Σcost += cost(a,y)/n
    end

    Σw2 = zero(T)
    for i = 2 : nn.L
        Σw2 += sum(nn.w[i].^2)
    end
    Σcost += (Σw2 * λ / 2n)
    Σcost
end





function simple()
    
    train_x = h5read("dnn/mnist.h5", "mnist/train_image")  #Float32 784 x 50000
    train_l = h5read("dnn/mnist.h5", "mnist/train_label")  #Int64 1 x 50000

    valid_x = h5read("dnn/mnist.h5", "mnist/test_image")  #Float32 784 x 10000
    valid_l = h5read("dnn/mnist.h5", "mnist/test_label")  #Int64 1 x 10000

    batch_valid = [(valid_x[:,i], valid_l[i]) for i = 1:length(valid_l)]
    batch_valid4cost = [(valid_x[:,i], onehot(Float32,10,valid_l[i])) for i = 1:length(valid_l)]


    nn = net{Float32}([784,100,10])
    epoch = 30
    η = 0.5f0
    λ = 5.0f0
    minibatchsize = 10
    minibatches = div(length(train_l), minibatchsize)
    
    for i = 1:epoch

        rng = MersenneTwister(i)
        batch_train = shuffle!(rng, [(train_x[:,i], onehot(Float32,10,train_l[i])) for i = 1:length(train_l)])

        for j = 0:minibatches-1
            update_minibatch(nn, batch_train[j*minibatchsize+1:(j+1)*minibatchsize], η, λ, length(train_l))
        end
        p1 = accu(nn, batch_valid)
        p2 = totalcost(nn, batch_train, λ)
        p3 = totalcost(nn, batch_valid4cost, λ)
        info("$p1    $p2    $p3")
    end
end



sigmoid(x) = 1.0 / (1.0 + exp(-x))
sigmoid_prime(x) = sigmoid(x) * (1.0 - sigmoid(x))

# ∂C/∂a for output activations
#return sum(nan_to_num(-y*log(a)-(1-y)*log(1-a)))
nan2num(x) = isnan(x) ? zero(typeof(x)) : x
cost(a::Array{T,1}, y::Array{T,1}) where T <: AbstractFloat = sum(nan2num.(-y.*log.(a) .- (one(T).-y).*log.(one(T).-a)))

∂C∂a(a, y) = a - y

function ∂C∂w!(∂C∂w, a, δ)
    for k = 1:size(∂C∂w,2)
        for j = 1:size(∂C∂w,1)
            ∂C∂w[j,k] = a[k] * δ[j]
        end
    end    
end