using JSON
using DataFrames
using WAV

#for i in pick(md, "/m/09l8g")["child_ids"]
#    println(pick(md,i)["name"])
#end

#mids = split(split(da[da[:,:x1].=="mkcMF8slW94",:x4][1])[1],",")


function mid(ontology)
    d = JSON.parsefile(ontology)
    t = Dict()
    for i in d
        t[i["name"]] = i["id"]
    end
    t
end


function norestriction(parsed)
    t = Dict()
    for i in parsed
        (i["restrictions"] == []) && (t[i["name"]] = i["id"])
    end
    t
end


function blacklist(parsed)
    t = Dict()
    for i in parsed
        (i["restrictions"] == ["blacklist"]) && (t[i["name"]] = i["id"])
    end
    t
end

function abstractlist(parsed)
    t = Dict()
    for i in parsed
        (i["restrictions"] == ["abstract"]) && (t[i["name"]] = i["id"])
    end
    t
end

function pick(parsed, m)
    t = Dict{String,Any}()
    for i in parsed
        (i["id"] == m) && (t = i)
    end
    t
end


function leaf(parsed, m)

    g = pick(parsed, m)
    r = Array{Any,1}()
    if isempty(g["child_ids"]) #leaf node
        push!(r, g["id"])
    else
        for i in g["child_ids"]
            r = [r; leaf(parsed,i)]
        end
    end
    r
end






# example: getfiles("D:\\Git\\Data\\", "m4a")    
function getfiles(path, filetype)
    
    list = Array{String,1}()
    for (root, dirs, files) in walkdir(path)
        #println("Directories in $root")
        for dir in dirs
            info(joinpath(root, dir)) # path to directories
        end
        #println("Files in $root")
        for file in files
            p = joinpath(root, file)
            file[end-length(filetype)+1:end] == filetype && push!(list, p)
            #println(p) # path to files
        end
    end
    list
end


function process(m4a, typewithdot, da, sane, sane_reverse, path_o)

    path_tmp = "D:\\tmp.wav"
    path_tmp16 = "D:\\tmp16.wav"
    
    h = length(m4a)
    m = length(typewithdot)
    count = 0
    for i in m4a
        # (a)
        run(`ffmpeg -y -i $i $path_tmp`)
        run(`sox $path_tmp -r 16000 $path_tmp16`)
        x,fs = wavread(path_tmp16)
        assert(fs == 16000.0f0)
        len = size(x,1)

        ytid = i[end-10-m:end-m]
        t0 = Int64(floor(16000 * da[da[:,:x1].==ytid,:x2][1]) + 1)
        t1 = Int64(floor(16000 * da[da[:,:x1].==ytid,:x3][1]))
        x = mean(x[t0:min(t1,len),:], 2)

        # (b)
        mids = split(split(da[da[:,:x1].==ytid,:x4][1])[1],",")
        for j in mids
            k = string(j[2],j[4:end])
            in(j, values(sane)) && wavwrite(x, joinpath(path_o, sane_reverse[j], "$k+$ytid.wav"), Fs=16000)
        end

        # update progress
        count += 1
        info("$count/$h complete")
    end
end



# flatten the tree data to flatten groups
# source: "F:\\GoogleAudioSet"
# sink: "E:\\GoogleAudioSet"
function flatten(source, sink)

    # I. get all files available for processing, files will be in form of for instance:
    #3-element Array{String,1}:
    #"D:\\Git\\julia-toolkit\\AudioSet\\data\\' A Narnia Lullaby ' on Ocarina-S051dpWiaeA.m4a"
    #"D:\\Git\\julia-toolkit\\AudioSet\\data\\- - Gears 2 Bullshit  - --mkcMF8slW94.m4a"
    #"D:\\Git\\julia-toolkit\\AudioSet\\data\\- Pavana el Todesco   - Renaissance italian music  ---7wUQP6G5EQ.m4a"
    m4a = getfiles(source, "m4a")
    webm = getfiles(source, "webm")


    # II. load "balanced_train_segments.csv" ans "eval_segments.csv" and merge to one table:
    #   42531×4 DataFrames.DataFrame
    #│ Row   │ x1            │ x2    │ x3    │ x4                                         │
    #├───────┼───────────────┼───────┼───────┼────────────────────────────────────────────┤
    #│ 1     │ "--PJHxphWEs" │ 30.0  │ 40.0  │ " /m/09x0r,/t/dd00088"                     │
    #│ 2     │ "--ZhevVpy1s" │ 50.0  │ 60.0  │ " /m/012xff"                               │
    #│ 3     │ "--aE2O5G5WE" │ 0.0   │ 10.0  │ " /m/03fwl,/m/04rlf,/m/09x0r"              │
    #│ 4     │ "--aO5cdqSAg" │ 30.0  │ 40.0  │ " /t/dd00003,/t/dd00005"                   │
    #⋮
    #│ 42527 │ "zyF8TGSRvns" │ 150.0 │ 160.0 │ " /m/0dwsp,/m/0dwtp,/m/0f8s22,/m/0j45pbj"  │
    #│ 42528 │ "zz35Va7tYmA" │ 30.0  │ 40.0  │ " /m/012f08,/m/07q2z82,/m/07qmpdm,/m/0k4j" │
    #│ 42529 │ "zzD_oVgzKMc" │ 30.0  │ 40.0  │ " /m/07pn_8q"                              │
    #│ 42530 │ "zzNdwF40ID8" │ 70.0  │ 80.0  │ " /m/04rlf,/m/0790c"                       │
    #│ 42531 │ "zzbTaK7CXJY" │ 30.0  │ 40.0  │ " /m/03m9d0z,/m/07qwyj0,/t/dd00092"        │
    da = [readtable("balanced_train_segments.csv", header=false, skipstart=3); readtable("eval_segments.csv", header=false, skipstart=3)]
    
    
    # III. get category infomation from "ontology.json"
    #      we only care categories of no restrictions -- neither abstract nor blacklisted
    #      these are the groups we want to create as the result of flatten
    #      we will get:
    #Dict{Any,Any} with 543 entries:
    #"Clickety-clack"                => "/m/07rwm0c"
    #"Caw"                           => "/m/07r5c2p"
    #"Dental drill, dentist's drill" => "/m/08j51y"
    #"Jazz"                          => "/m/03_d0"
    #"Crunch"                        => "/m/07s12q4"
    #"House music"                   => "/m/03mb9"
    #"Music for children"            => "/m/05fw6t"
    #"Yip"                           => "/m/07r_k2n"
    #"Jet engine"                    => "/m/04229"
    #"Walk, footsteps"               => "/m/07pbtc8"
    #"Ratchet, pawl"                 => "/m/02bm9n"
    #⋮                               => ⋮
    sane = norestriction(JSON.parsefile("ontology.json"))
    sane_reverse = Dict(value => key for (key, value) in sane)
    path_o = sink
    for i in keys(sane)
        rm(joinpath(path_o,i), force=true, recursive=true)
        mkpath(joinpath(path_o,i))
    end


    # IV. iterate over m4a/webm files:
    #     (a) locate time duration and extract pcm tracks to wav
    #     (b) get the file YTID from file name -> lookup in da for MIDs -> for each MID,
    #         use sane for group name
    process(m4a, ".m4a", da, sane, sane_reverse, path_o)
    process(webm, ".webm", da, sane, sane_reverse, path_o)
    
    info("end of processing.")
end
