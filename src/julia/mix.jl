# management of wav files for machine learning projects
# lixun.xia@outlook.com
# 2017-10-16
using SHA
using WAV
using JSON
using HDF5

include("audiofeatures.jl")
include("forward.jl")
include("indicator.jl")




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
    p = indicator(10)

    for (i,j) in enumerate(list)
        d += open(j) do f
            sha256(f)
            end
        update(p, i, n)
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
function specification_gen()
    
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

        "feature" => Dict("frame_size"=>512,"step_size"=>128,"window"=>"Hamming","frame_neighbour"=>5,"nat_size"=>7)
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
    p = indicator(10)
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
        update(p, i, n)
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
    mr = s["mix_range"]         #[0.1, 0.9]

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
    output = joinpath(s["mix_root"],s["hdf5"])
    rm(output, force=true)


    # feature specification
    m = s["feature"]["frame_size"]
    d = s["feature"]["step_size"]
    param = Frame1D{Int64}(s["sample_rate"], m, d, 0)
    win = Dict("Hamming"=>hamming, "Hann"=>hann)

    # process each mixed to spectral domain
    srand(s["seed"])
    progress = indicator(10)

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

        x_speech = g[j][1] .* repeat(x_speech, outer=speech_dup)
        x_noise .*= g[j][2]
        x_noise_ = 1e-7*randn(size(x_speech))

        assert(length(a[j]) == noise_dup)
        for k in a[j]
            x_noise_[k[1]:k[2]] = x_noise
        end
        
        # for verification purpose
        # h5write(output, "$j/speech", x_speech)
        # h5write(output, "$j/noise", x_noise_)
        # h5write(output, "$j/mix", x_mix)
        # wavwrite([x r], i*"label",Fs=s["sample_rate"])
        f = win[s["feature"]["window"]]
        pxx_speech, lu = power_spectrum(x_speech, param, m, window=f)
        pxx_noise, lu = power_spectrum(x_noise_, param, m, window=f)
        pxx_mix, lu = power_spectrum(x_mix, param, m, window=f)
        
        h5write(output, "$j/speech", log.(pxx_speech.+eps()))
        h5write(output, "$j/noise", log.(pxx_noise.+eps()))
        h5write(output, "$j/mix", log.(pxx_mix.+eps()))
        # "frame_size"/2+1 by N matrix
        update(progress, i, length(a))
    end
    info("feature written to $(output)")
end







# global variance:
# remove old global.h5 and make new
function statistics(specification)

    # read the specification for feature extraction
    s = JSON.parsefile(specification)
    pathstat = joinpath(s["mix_root"],"global.h5")
    rm(pathstat, force=true)
    m = div(s["feature"]["frame_size"], 2) + 1

    fid = h5open(joinpath(s["mix_root"],s["hdf5"]),"r")
    l = length(names(fid))

    # get global frame count
    n = zero(Int128)                            
    progress = indicator(10)
    for (i,j) in enumerate(names(fid))   
        n += size(read(fid[j]["mix"]), 2)        
        update(progress, i, l)
    end
    info("global spectrum count: $n")

    # get global mean log power spectrum
    μ = zeros(m)
    σ = zeros(m)
    μi = zeros(BigFloat, m, l)

    rewind(progress)
    for(i,j) in enumerate(names(fid))
        x = read(fid[j]["mix"])                 
        for k = 1:m
            μi[k,i] = sum_kbn(view(x,k,:))
        end
        update(progress, i, l)
    end
    for k = 1:m
        μ[k] = sum_kbn(view(μi,k,:))/n
    end
    info("global spectrum μ (dimentionless): $(mean(μ))")

    # get global std for unit variance
    fill!(μi, zero(BigFloat))
    rewind(progress)
    for(i,j) in enumerate(names(fid))
        x = read(fid[j]["mix"])              
        for k = 1:m
            μi[k,i] = sum_kbn((view(x,k,:)-μ[k]).^2)
        end
        update(progress, i, l)
    end
    for k = 1:m
        σ[k] = sqrt(sum_kbn(view(μi,k,:))/(n-1))
    end
    info("global spectrum σ (dimentionless): $(mean(σ))")

    h5write(pathstat, "mu", μ)
    h5write(pathstat, "std", σ)
    h5write(pathstat, "frames", Int64(n))
    assert(n < typemax(Int64))
    info("global results written to $(pathstat)")

    close(fid)
end




#######line of pure graceful and joy#######
# specification.json
# partitions: number of .h5's as output
function tensor(specification, partitions::Int64, specification_train)

    # read the specification for feature extraction
    s = JSON.parsefile(specification) 
    m = div(s["feature"]["frame_size"], 2) + 1
    radius = s["feature"]["frame_neighbour"]
    nat = s["feature"]["nat_size"]
    periph = (2radius+2) * m

    # extract global stat info
    # note that the global stat must be of training
    s_train = JSON.parsefile(specification_train)
    stat = joinpath(s_train["mix_root"],"global.h5")
    #n = h5read(stat,"frames")
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
            nf[j] = size(read(fid[groups[i*np+j]]["mix"]), 2)
        end  
        data = zeros(periph, sum(nf))
        label = zeros(m, sum(nf))

        # fill in each partition with context data and nat data
        fin = cumsum(nf)
        start = vcat(1, 1+fin[1:end-1])
        progress = indicator(10)

        for j = 1:np
            tmp = read(fid[groups[i*np+j]]["mix"])
            tmp = (tmp.-μ)./σ
            data[:, start[j]:fin[j]] = sliding(tmp, radius, nat)

            tmp = read(fid[groups[i*np+j]]["speech"])
            label[:, start[j]:fin[j]] = (tmp.-μ)./σ
            update(progress, j, np)
        end
        pathout = joinpath(s["mix_root"], "tensor-$i.h5")
        h5write(pathout, "data", Float32.(data))
        h5write(pathout, "label", Float32.(label))
        info("partition $i ok")
    end

    close(fid)
end








function main()

    valid_spec = "D:\\4-Workspace\\mix\\valid\\specification-2017-11-13T16-50-41-801.json"
    valid_lab = "D:\\4-Workspace\\mix\\valid\\label.json"
    valid_gain = "D:\\4-Workspace\\mix\\valid\\gain.json"

    train_spec = "D:\\4-Workspace\\mix\\train\\specification-2017-11-13T16-50-41-801.json"
    train_lab = "D:\\4-Workspace\\mix\\train\\label.json"
    train_gain = "D:\\4-Workspace\\mix\\train\\gain.json"

    # mix(train_spec)                                  # generate mixed wav with labelings and gains
    # feature(train_spec, train_lab, train_gain)       # extract plain features, to valid.h5/train.h5
    # statistics(train_spec)                                # find out the global stats: mean/std/total frames
    # tensor(train_spec, 10, train_spec)                           # convert plain features to tensor input    

    mix(valid_spec)                                  # generate mixed wav with labelings and gains
    feature(valid_spec, valid_lab, valid_gain)       # extract plain features, to valid.h5/train.h5
    statistics(valid_spec)                                # find out the global stats: mean/std/total frames
    tensor(valid_spec, 1, train_spec)                           # convert plain features to tensor input
end