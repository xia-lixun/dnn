using MAT
include("indicator.jl")




struct tf{T <: AbstractFloat}

    w::Array{Array{T,2},1}  # weights, w[1] is not used
    b::Array{Array{T,1},1}  # biases, b[1] is not used
    L::Int64  # layers
    width_i::Int64
    width_o::Int64

    function tf{T}(model::String) where T <: AbstractFloat
        nn = matread(model)
        w = Array{Array{T,2},1}()
        b = Array{Array{T,1},1}()

        L = 4
        for i = 1:L
            push!(w, transpose(nn["W$i"]))
            push!(b, vec(nn["b$i"]))
        end

        wdi = size(w[1], 2)
        wdo = size(w[end], 1)

        new(w, b, L, wdi, wdo)
    end 
end


sigmoid(x::T) where T <: AbstractFloat = one(T) / (one(T) + exp(-x))


# Propagate the input data matrix through neural net
# 1. x is column major, i.e. each column is an input vector to the net 
function forward(model, x::Array{T,2}) where T <: AbstractFloat
    
    nn = tf{Float32}(model)
    y = zeros(T, nn.width_o, size(x,2))

    for i = 1:size(x,2)
        a = sigmoid.(nn.w[1] * view(x,:,i) .+ nn.b[j])
        for j = 2 : nn.L-1
            a .= sigmoid.(nn.w[j] * a .+ nn.b[j])
        end
        y[:,i] .= nn.w[nn.L] * a .+ nn.b[nn.L]
    end
    y
end


# do magnitude processing through the net
# 1. input is un-normalized col-major magnitude spectrum
# 2. output is un-normalized col-major noise-reduced magnitude spectrum
function magnitude_processing!(model::String, x::Array{T,2}, r::Int64, μ::Array{T,1}, σ::Array{T,1}) where T <: AbstractFloat
    x .= (x .- μ) ./ σ
    y = sliding(x, r)
    nn = tf{Float32}(model)
    x .= forward(nn, y) .* σ .+ μ
end


# Get frame context from spectrogram x with radius r
# 1. x must be col major, i.e. each col is a spectrum frame for example, 257 x L matrix
# 2. y will be (257*(neighbour*2+1+nat)) x L
# 3. todo: remove allocations for better performance
symm(i,r) = i-r:i+r

function sliding(x::Array{T,2}, r::Int64) where T <: AbstractFloat

    m, n = size(x)
    head = repmat(x[:,1], 1, r)
    tail = repmat(x[:,end], 1, r)
    x = hcat(head, x, tail)
    y = zeros((2r+2)*m, n)

    for i = 1:n
        focus = view(x,:,symm(r+i,r))
        nat = sum(focus, 2) / (2r+1)
        y[:,i] = vec(hcat(focus,nat))
    end
    y
end








