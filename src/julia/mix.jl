# management of wav files for machine learning projects
# lixun.xia@outlook.com
# 2017-10-16
using SHA
using WAV
using JSON
using HDF5
include("audiofeatures.jl")




#filst(path) will list all busfolders under path, without root!
#flist(path, t=".wav") will list all wav files under path and uid 
function flist(path; t="")

    x = Array{String,1}()
    for (root, dirs, files) in walkdir(path)
        for dir in dirs
            isempty(t) && push!(x, dir)
        end
        for file in files
            !isempty(t) && lowercase(file[end-length(t)+1:end]) == lowercase(t) && push!(x, joinpath(root, file))
        end
    end
    x
end


function fsha256(list)

    d = zeros(UInt8,32)
    n = length(list)
    pz = -1
    for (i,j) in enumerate(list)
        d += open(j) do f
            sha256(f)
            end
        p = Int64(round((i/n)*100))
        in(p, 0:10:100) && (p != pz) && (pz = p; print("."))
    end
    d
end




# api
function updatesha256(path)

    p = joinpath(path,"index.sha256")
    writedlm(p, fsha256(flist(path, t=".wav")))
    info("checksum updated in $p")
end
function checksha256(path)

    p = readdlm(joinpath(path,"index.sha256"), UInt8)[:,1]
    q = fsha256(flist(path, t="wav"))
    ok = (0x0 == sum(p-q))
end




# 1. walk through path_in folder for all wav files recursively
# 2. convert to target fs
# 3. put result to path_out foler linearly
function resample(path_in, path_out, target_fs)

    a = flist(path_in, t=".wav")
    n = length(a)
    u = Array{Int64,1}(n)

    for (i,j) in enumerate(a)
        run(`ffmpeg -y -i $j D:\\temp.wav`)
        p = joinpath(path_out, relpath(dirname(j), path_in))
        mkpath(p)
        p = joinpath(p, basename(j))
        run(`sox D:\\temp.wav -r 16000 $p`)
            
        x,fs = wavread(p)
        assert(fs == 16000.0f0)
        u[i] = size(x,1)
        
        info("$i/$n complete")
    end
    info("max: $(maximum(u)/16000) seconds")
    info("min: $(minimum(u)/16000) seconds")
    rm("D:\\temp.wav", force=true)
end



#generate template JSON file based on folder contents
function specgen()
    
    x = Array{Dict{String,Any},1}()
    a = Dict(
        "seed" => 1234,
        "sample_rate" => 16000,
        "sample_space"=>17,
        "mix_snr" => [20.0, 15.0, 10.0, 5.0, 0.0, -5.0],
        "speech_level" => [-22.0, -32.0, -42.0],
        "mix_base" => "speech",
        "mix_range" => [0.1, 0.9],
        "mix_root" => "D:\\mix-utility\\mixed",
        "hdf5" => "train.h5",
        "speech_root" => "D:\\VoiceBox\\TIMIT-16k\\train",
        "noise_root" => "D:\\NoiseBox\\104Nonspeech-16k",
        "build_speechlevel_index" => true,
        "build_noiselevel_index" => true,
        "noise_class" => x,

        "feature" => Dict("frame_size"=>512,"step_size"=>128,"window"=>"Hamming","frame_neighbour"=>6,"frame_noise_est"=>7)
        )
    for i in flist(a["noise_root"])
        push!(x, Dict("name"=>i,"type"=>"stationary|nonstationary|impulsive","percent"=>0.0))
    end

    #rm(a["mixoutput"], force=true, recursive=true)
    !isdir(a["mix_root"]) && mkpath(a["mix_root"])
    open(joinpath(a["mix_root"],"specification-$(replace(replace("$(now())",":","-"),".","-")).json"),"w") do f
        write(f, JSON.json(a))
    end
end




rms(x,dim) = sqrt.(sum((x.-mean(x,dim)).^2,dim)/size(x,dim))
rms(x) = sqrt(sum((x-mean(x)).^2)/length(x))

# calculate index for noise samples:
# peak level: (a) as level of impulsive sounds (b) avoid level clipping in mixing
# rms level: level of stationary sounds
# median level: level of non-stationary sounds
#
# note: wav read errors will be skipped but warning pops up
#       corresponding features remain zero
function build_level_index(path, rate)

    a = flist(path, t=".wav")
    n = length(a)
    lpeak = zeros(n)
    lrms = zeros(n)
    lmedian = zeros(n)
    len = zeros(Int64, n)
    
    #uid = Array{String,1}(n)
    m = length(path)

    # wav must be monochannel and fs==16000
    pz = -1
    for (i,j) in enumerate(a)
        try
            x, fs = wavread(j)
            assert(Int64(fs) == Int64(rate))
            assert(size(x,2) == 1)
            x = x[:,1]
            lpeak[i] = maximum(abs.(x))
            lrms[i] = rms(x)
            lmedian[i] = median(abs.(x))
            len[i] = length(x)
            #uid[i] = replace(j[m+1:end-length(".wav")], "/", "+")
        catch
            warn(j)
        end

        p = Int64(round((i/n)*100))
        in(p, 0:10:100) && (p != pz) && (pz = p; print("."))
        
    end

    # save level index to csv
    index = joinpath(path, "index.level")
    writedlm(index, [a lpeak lrms lmedian len], ',')
    info("index build to $index")
end



# mix procedure implements the specification
function mix(specification)

    # read the specification for mixing task
    s = JSON.parsefile(specification)

    fs = s["sample_rate"]          #16000
    n = s["sample_space"]          #17
    snr = s["mix_snr"]          #[-20.0, -10.0, 0.0, 10.0, 20.0]
    spl = s["speech_level"]     #[-22.0, -32.0, -42.0]
    mr = s["mix_range"]         #[0.1, 0.6]

    # remove existing wav file in the mix folder
    clean = flist(s["mix_root"], t=".wav")
    for i in clean
        rm(i, force=true)
    end
    mkpath(joinpath(s["mix_root"],"wav"))

    #Part I. Noise treatment
    !isdir(s["noise_root"]) && error("noise root doesn't exist")
    if s["build_noiselevel_index"]
        for i in s["noise_class"]
            build_level_index(joinpath(s["noise_root"],i["name"]), fs)
            updatesha256(joinpath(s["noise_root"],i["name"]))
        end
    else
        for i in s["noise_class"]
            !checksha256(joinpath(s["noise_root"],i["name"])) && error("checksum $(i["name"])")
            info("noise checksum ok")
        end
    end
    #index format: path-to-wav-file, peak-level, rms-level, median, length-in-samples
    ni = Dict( x["name"] => readdlm(joinpath(s["noise_root"], x["name"],"index.level"), ',') for x in s["noise_class"])


    #Part II. Speech treatment
    !isdir(s["speech_root"]) && error("speech root doesn't exist")
    if s["build_speechlevel_index"]
        #build_level_index(s["speech_rootpath"])
        #assume index ready by Matlab:activlevg() and provided as csv in format: path-to-wavfile, speech-level, length-in-samples
        updatesha256(s["speech_root"])
    else
        !checksha256(s["speech_root"]) && error("speech data checksum error")
        info("speech checksum ok")
    end
    #index format: path-to-wav-file, peak-level, speech-level(dB), length-in-samples
    si = readdlm(joinpath(s["speech_root"],"index.level"), ',', header=false, skipstart=3)
    #(si, ni)


    # Part III. Mixing them up
    fcount = 1
    label = Dict{String, Array{Tuple{Int64, Int64},1}}()
    gain = Dict{String, Array{Float64,1}}()
    srand(s["seed"])

    for i in s["noise_class"]
        for j = 1:Int64(round(i["percent"] * 0.01 * n)) # items in each noise category
            
            gvec = zeros(2)

            # 3.0 preparation of randomness
            sp = spl[rand(1:length(spl))]
            sn = snr[rand(1:length(snr))]
            rs = rand(1:size(si,1))
            rn = rand(1:size(ni[i["name"]],1))

            # 3.1: random speech, in x[:,1]
            x = Array{Float64,1}()
            try
                x = wavread(si[rs,1])[1][:,1]
            catch
                #todo: test if my wav class could handle the corner cases of WAV.jl
                #todo: wrap my wav class with c routines to wav.dll, then wrap with julia
                warn("missing $(si[rs,1])")
            end
            g = 10^((sp-si[rs,3])/20)
            g * si[rs,2] > 1 && (g = 1 / si[rs,2]; info("relax gain to avoid clipping $(si[rs,1]):$(si[rs,3])->$(sp)(dB)"))
            x *= g #level speech to target level
            gvec[1] = g
            
            # 3.2: random noise
            u = Array{Float64,1}()
            try
                u = wavread(ni[i["name"]][rn,1])[1][:,1]
            catch
                #todo: test if my wav class could handle the corner cases of WAV.jl
                #todo: wrap my wav class with c routines to wav.dll, then wrap with julia
                warn("missing $(ni[i["name"]][rn,1])")
            end

            #3.3: random snr, calculate noise level based on speech level and snr
            t = 10^((sp-sn)/20)
            if i["type"] == "impulsive" 
                g = t / ni[i["name"]][rn,2]
            elseif i["type"] == "stationary"
                g = t / ni[i["name"]][rn,3]
            elseif i["type"] == "nonstationary"
                g = t / ni[i["name"]][rn,4]
            else
                error("wrong type in $(i["name"])")
            end
            g * ni[i["name"]][rn,2] > 1 && (g = 1 / ni[i["name"]][rn,2]; info("relax gain to avoid clipping $(ni[i["name"]][rn,1])"))
            u *= g
            gvec[2] = g
            
            #3.4: portion check
            nid = replace(relpath(ni[i["name"]][rn,1],s["noise_root"]), "\\", "+")[1:end-4]
            sid = replace(relpath(si[rs,1],s["speech_root"]), "\\", "+")[1:end-4]
            
            p = si[rs,4]
            q = ni[i["name"]][rn,5]
            if lowercase(s["mix_base"]) == "speech" 
                (x, u) = (u, x)
                p = ni[i["name"]][rn,5]
                q = si[rs,4]
            end
            η = p/q
            # x,p is the shorter signal
            # u,q is the longer signal

            if mr[1] <= η <= mr[2]
                rd = rand(1:q-p)
                u[rd:rd+p-1] += x
                # clipping sample if over-range?
                path = joinpath(s["mix_root"],"wav","$(fcount)+$(nid)+$(sid)+1+1+$(sp)+$(sn).wav")
                wavwrite(u, path, Fs=fs)
                label[path] = [(rd, rd+p-1)]
                gain[path] = gvec
            # η > mr[2] or η < mr[1]    
            else 
                np = 1
                nq = 1
                while !(mr[1] <= η <= mr[2])
                    η > mr[2] && (nq += 1)
                    η < mr[1] && (np += 1)
                    η = (np*p)/(nq*q)                    
                end
                path = joinpath(s["mix_root"],"wav","$(fcount)+$(nid)+$(sid)+$(np)+$(nq)+$(sp)+$(sn).wav")
                stamp = Array{Tuple{Int64, Int64},1}()

                u = repeat(u, outer=nq)
                pp = Int64(floor((nq*q)/np)) 
                for k = 0:np-1
                    rd = k*pp+rand(1:pp-p)
                    u[rd:rd+p-1] += x
                    push!(stamp,(rd, rd+p-1))
                end
                wavwrite(u, path, Fs=fs)
                label[path] = stamp
                gain[path] = gvec
            end
            fcount += 1
            info("η = $η")
        end
    end
    open(joinpath(s["mix_root"],"label.json"),"w") do f
        write(f, JSON.json(label))
    end
    open(joinpath(s["mix_root"],"gain.json"),"w") do f
        write(f, JSON.json(gain))
    end    
    info("label written to $(joinpath(s["mix_root"],"label.json"))")
    info("gain written to $(joinpath(s["mix_root"],"gain.json"))")
end







# remove old feature.h5 and make new
function feature(specification, label, gain)

    # mixed file and label info
    a = JSON.parsefile(label)
    g = JSON.parsefile(gain)
    s = JSON.parsefile(specification)
    assert(s["sample_space"] == length(a))
    
    # remove existing .h5 training/valid/test data
    rm(joinpath(s["mix_root"],s["hdf5"]), force=true)


    # feature specification
    m = s["feature"]["frame_size"]
    d = s["feature"]["step_size"]
    param = Frame1D{Int64}(s["sample_rate"], m, d, 0)
    win = Dict("Hamming"=>hamming, "Hann"=>hann)

    # process each mixed to spectral domain
    srand(s["seed"])
    ptz = -1
    for (i,j) in enumerate(keys(a))
        
        p = split(j[1:end-length(".wav")],"+")
        #[1]"D:\\mix-utility\\mixed\\1"
        #[2]"impulsive"
        #[3]"n48"
        #[4]"dr1"      
        #[5]"mklw0"    
        #[6]"sa1"
        #[7]"2"
        #[8]"3"      
        #[9]"-22.0"    
        #[10]"20.0" 
        noise_dup = parse(Int64,p[7])
        speech_dup = parse(Int64,p[8])

        x_mix = wavread(j)[1][:,1]
        x_noise = wavread(joinpath(s["noise_root"],p[2],p[3]) * ".wav")[1][:,1]
        x_speech = wavread(joinpath(s["speech_root"],p[4],p[5],p[6]) * ".wav")[1][:,1]

        #g = 10^((tagspl-refspl)/20)
        #g * refpk > 1 && (g = 1 / refpk; info("relax gain to avoid clipping $(refpk):$(refspl)->$(tagspl)(dB)"))
        #r = g * repeat(r, outer=dup) #level speech to target level
        x_speech = g[j][1] .* repeat(x_speech, outer=speech_dup)
        x_noise .*= g[j][2]
        x_noise_ = 1e-7*randn(size(x_speech))
        for k in a[j]
            x_noise_[k[1]:k[2]] = x_noise
        end
        
        #h5write(joinpath(s["mix_root"],s["hdf5"]), "$j/speech", x_speech)
        #h5write(joinpath(s["mix_root"],s["hdf5"]), "$j/noise", x_noise_)
        #h5write(joinpath(s["mix_root"],s["hdf5"]), "$j/mix", x_mix)

        #wavwrite([x r], i*"label",Fs=s["sample_rate"])
        pxx_speech = power_spectrum(x_speech, param, m, window=win[s["feature"]["window"]])
        pxx_noise = power_spectrum(x_noise_, param, m, window=win[s["feature"]["window"]])
        pxx_mix = power_spectrum(x_mix, param, m, window=win[s["feature"]["window"]])
        
        h5write(joinpath(s["mix_root"],s["hdf5"]), "$j/speech", transpose(log.(pxx_speech.+eps())))
        h5write(joinpath(s["mix_root"],s["hdf5"]), "$j/noise", transpose(log.(pxx_noise.+eps())))
        h5write(joinpath(s["mix_root"],s["hdf5"]), "$j/mix", transpose(log.(pxx_mix.+eps())))
        # N x s["feature"]["frame_size"]/2+1 matrix

        pt = Int64(round((i/length(a))*100))
        in(pt, 0:10:100) && (pt != ptz) && (ptz = pt; print("."))
    end
    info("feature written to $(joinpath(s["mix_root"],s["hdf5"]))")
end


# global variance:
# remove old global.h5 and make new
function gstat(specification)

    # read the specification for feature extraction
    s = JSON.parsefile(specification) 
    rm(joinpath(s["mix_root"],"global.h5"), force=true)

    m = div(s["feature"]["frame_size"], 2) + 1 # m = 257
    fid = h5open(joinpath(s["mix_root"],s["hdf5"]),"r")
    l = length(names(fid))

    # get global frame count
    n = zero(Int128)                            
    ptz = -1
    for (i,j) in enumerate(names(fid))   
        n += size(read(fid[j]["mix"]),1)        
        pt = Int64(round(100 * (i/l)))
        in(pt, 0:10:100) && (pt != ptz) && (ptz = pt; print("."))
    end
    info("global spectrum count: $n")

    # get global mean log power spectrum
    μ = zeros(Float64, m)
    μi = zeros(BigFloat, l, m)

    ptz = -1
    for(i,j) in enumerate(names(fid))
        x = read(fid[j]["mix"])                 
        for k = 1:m
            μi[i,k] = sum_kbn(x[:,k])
        end
        pt = Int64(round(100 * (i/l)))        
        in(pt, 0:10:100) && (pt != ptz) && (ptz = pt; print("."))        
    end
    for k = 1:m
        μ[k] = Float64(sum_kbn(μi[:,k])/n)
    end
    info("global spectrum μ (dimentionless): $(mean(μ))")


    # get global std for unit variance
    σ = zeros(Float64, m)
    fill!(μi, zero(BigFloat))
    ptz = -1
    for(i,j) in enumerate(names(fid))
        x = read(fid[j]["mix"])              
        for k = 1:m
            μi[i,k] = sum_kbn((x[:,k]-μ[k]).^2)
        end
        pt = Int64(round(100 * (i/l)))        
        in(pt, 0:10:100) && (pt != ptz) && (ptz = pt; print("."))        
    end
    for k = 1:m
        σ[k] = Float64(sqrt(sum_kbn(μi[:,k])/(n-1)))
    end
    info("global spectrum σ (dimentionless): $(mean(σ))")

    h5write(joinpath(s["mix_root"],"global.h5"), "mu", μ)
    h5write(joinpath(s["mix_root"],"global.h5"), "std", σ)
    h5write(joinpath(s["mix_root"],"global.h5"), "frames", Int64(n))
    assert(n < typemax(Int64))
    info("global results written to $(joinpath(s["mix_root"],"global.h5"))")

    close(fid)
end




#######line of pure graceful and joy#######
# specification.json
# partitions: number of .h5's as output
function context(specification, partitions::Int64)

    # read the specification for feature extraction
    s = JSON.parsefile(specification) 
    m = div(s["feature"]["frame_size"], 2) + 1 # m = 257
    radius = s["feature"]["frame_neighbour"]
    periph = (2radius+2) * m

    # extract global stat info
    stat = joinpath(s["mix_root"],"global.h5")
    n = h5read(stat,"frames")
    μ = h5read(stat,"mu")
    σ = h5read(stat,"std")

    # context processing
    fid = h5open(joinpath(s["mix_root"],s["hdf5"]),"r")
    groups = names(fid)
    np = div(length(groups), partitions)  # groups per partitions

    
    for i = 0:partitions-1

        # find out the size of each partition
        nf = zeros(Int64, np)
        for j = 1:np
            nf[j] = size(read(fid[groups[i*np+j]]["mix"]),1)
        end  
        data = zeros(sum(nf), periph)
        label = zeros(sum(nf), m)

        # fill in each partition with context data and nat data
        fin = cumsum(nf)
        start = vcat(1, 1+fin[1:end-1])
        ptz = -1
        for j = 1:np
            data[start[j]:fin[j],:] = sliding!(read(fid[groups[i*np+j]]["mix"]), radius, μ, σ)        
            label[start[j]:fin[j],:] = (read(fid[groups[i*np+j]]["speech"]) .- (μ')) ./ (σ')

            # update progress
            pt = Int64(round(100 * (j/np)))
            in(pt, 0:10:100) && (pt != ptz) && (ptz = pt; print("."))
        end
        pathout = joinpath(s["mix_root"], "tensor-$i.h5")
        h5write(pathout, "data", data)
        h5write(pathout, "label", label)
        info("partition $i ok")
    end

    close(fid)
end

# x = L x 257 matrix
# return y: L x (257*(neighbour*2+1+1))
symm(i,r) = i-r:i+r
function sliding!(x::Array{Float64,2}, r::Int64, μ::Array{Float64,1}, σ::Array{Float64,1})

    # normalize
    x = (x.-(μ'))./(σ')

    # get sliding frame context
    m, n = size(x)
    head = repmat(x[1,:]', r, 1)
    tail = repmat(x[end,:]', r, 1)
    x = vcat(head, x, tail)

    y = zeros(m,(2r+2)*n)
    for i = 1:m
        focus = x[symm(r+i,r),:]
        nat = sum(focus,1)[1,:] / (2r+1)
        y[i,:] = vec(hcat(transpose(focus),nat))
    end
    y
end







function datagen()
    valid_spec = "D:\\4-Workspace\\mix\\valid\\specification-2017-11-13T16-50-41-801.json"
    valid_lab = "D:\\4-Workspace\\mix\\valid\\label.json"
    valid_glob = "D:\\4-Workspace\\mix\\valid\\global.h5"
    valid_gain = "D:\\4-Workspace\\mix\\valid\\gain.json"

    train_spec = "D:\\4-Workspace\\mix\\train\\specification-2017-11-13T16-50-41-801.json"
    train_lab = "D:\\4-Workspace\\mix\\train\\label.json"
    train_glob = "D:\\4-Workspace\\mix\\train\\global.h5"
    train_gain = "D:\\4-Workspace\\mix\\train\\gain.json"

    mix(train_spec)                                  # generate mixed wav with labelings and gains
    feature(train_spec, train_lab, train_gain)       # extract plain features, to valid.h5/train.h5
    gstat(train_spec)                                # find out the global stats: mean/std/total frames
    context(train_spec, 10)                           # convert plain features to tensor input    

    mix(valid_spec)                                  # generate mixed wav with labelings and gains
    feature(valid_spec, valid_lab, valid_gain)       # extract plain features, to valid.h5/train.h5
    gstat(valid_spec)                                # find out the global stats: mean/std/total frames
    context(valid_spec, 1)                           # convert plain features to tensor input
end