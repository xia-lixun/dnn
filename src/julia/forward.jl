using MAT
using HDF5
using JSON
using WAV

include("indicator.jl")
include("audiofeatures.jl")



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
function forward(model, x::AbstractArray{T,2}) where T <: AbstractFloat
    
    nn = tf{Float32}(model)
    n = size(x,2)
    y = zeros(T, nn.width_o, n)

    p = indicator(10)
    for i = 1:n
        a = sigmoid.(nn.w[1] * view(x,:,i) .+ nn.b[1])
        for j = 2 : nn.L-1
            a .= sigmoid.(nn.w[j] * a .+ nn.b[j])
        end
        y[:,i] .= nn.w[nn.L] * a .+ nn.b[nn.L]
        update(p, i, n)
    end
    info("nn feed forward done.")
    y
end


# do magnitude processing through the net
# 1. input is un-normalized col-major magnitude spectrum
# 2. output is un-normalized col-major noise-reduced magnitude spectrum
function psd_processing!(model::String, 
                         x::AbstractArray{T,2}, 
                         r::Int64,
                         t::Int64,
                         μ::AbstractArray{T,1}, σ::AbstractArray{T,1}) where T <: AbstractFloat
    x .= log.(x .+ eps())
    x .= (x .- μ) ./ σ
    y = sliding(x, r, t)
    nn = tf{Float32}(model)
    x .= forward(nn, y) .* σ .+ μ
    x .= exp.(x)
end


# do COLA processing of a wav file
function cola_processing(specification::String, wav::String; model::String = "")

    s = JSON.parsefile(specification)
    s_frame = s["feature"]["frame_size"]
    s_hop = s["feature"]["step_size"]
    s_r = s["feature"]["frame_neighbour"]
    s_t = s["feature"]["nat_size"]
    s_fs = s["sample_rate"]
    s_win = Dict("Hamming"=>hamming, "Hann"=>hann)

    # get global mu and std
    stat = joinpath(s["mix_root"], "global.h5")
    μ = Float32.(h5read(stat, "mu"))
    σ = Float32.(h5read(stat, "std"))

    # get input data
    x, fs = wavread(wav)
    assert(fs == typeof(fs)(s_fs))
    x = Float32.(x)
    
    # convert to frequency domain
    param = Frame1D{Int64}(s_fs, s_frame, s_hop, 0)
    nfft = s_frame
    ρ = Float32(1 / nfft)
    _cola = s_hop / sum(hamming(Float32, nfft))
    
    𝕏, lu = spectrogram(view(x,:,1), param, nfft, window=s_win[s["feature"]["window"]])
    m = size(𝕏, 2)
    y = zeros(Float32, lu)

    # reconstruct
    ImagAssert = 0.0f0
    if isempty(model)
        𝕏 = _cola * real(ifft(𝕏, 1))
    else
        # keep phase info
        𝚽 = angle.(𝕏)

        # calculate power spectra
        nfft2 = div(nfft,2)+1
        ℙ = ρ.*(abs.(view(𝕏,1:nfft2,:))).^2
        psd_processing!(model, ℙ, s_r, s_t, μ, σ)
        ℙ .= sqrt.(ℙ)./ρ
        ℙ = vcat(ℙ, ℙ[end-1:-1:2,:])
        𝕏 = ifft(ℙ .* exp.(𝚽 .* im), 1)
        ImagAssert = sum(imag(𝕏))
        𝕏 = _cola * real(𝕏)
    end

    for k = 0:m-1
        y[k*s_hop+1 : k*s_hop+nfft] .+= 𝕏[:,k+1]
    end
    wavwrite(y, wav[1:end-4]*"-processed.wav", Fs=s_fs)
    ImagAssert
end
















# Get frame context from spectrogram x with radius r
# 1. x must be col major, i.e. each col is a spectrum frame for example, 257 x L matrix
# 2. y will be (257*(neighbour*2+1+nat)) x L
# 3. todo: remove allocations for better performance
symm(i,r) = i-r:i+r

# r: radius
# t: noise estimation frames
function sliding(x::Array{T,2}, r::Int64, t::Int64) where T <: AbstractFloat

    m, n = size(x)
    head = repmat(x[:,1], 1, r)
    tail = repmat(x[:,end], 1, r)
    x = hcat(head, x, tail)
    y = zeros((2r+2)*m, n)

    for i = 1:n
        focus = view(x,:,symm(r+i,r))
        nat = sum(view(focus,:,1:t), 2) / t
        y[:,i] = vec(hcat(focus,nat))
    end
    y
end








