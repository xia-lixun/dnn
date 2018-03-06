import JSON

db = ["/oak/speech_denoise/Voice/TIMIT+LP7/TIMIT", 
      "/oak/speech_denoise/Voice/TIMIT+LP7/LP7"]

sr = 16000
len_max = 0
len_min = 2^31-1
len_sum = 0

mg = Dict{String, Any}()
mg["DATA"] = Dict{String, Any}()
mg["META"] = Dict{String, Any}()

for i in db
    d = JSON.parsefile(joinpath(i,"level.json"))
    
    d["META"]["len_max"] > len_max && (len_max = d["META"]["len_max"])
    d["META"]["len_min"] < len_min && (len_min = d["META"]["len_min"])
    len_sum += d["META"]["len_sum"]
    assert(sr == d["META"]["sample_rate"])

    for j in d["DATA"]
        mg["DATA"][basename(i)*"/"*j[1]] = j[2]
    end
end
mg["META"]["sample_rate"] = sr
mg["META"]["len_max"] = len_max
mg["META"]["len_min"] = len_min
mg["META"]["len_sum"] = len_sum

open(joinpath(dirname(db[1]),"level.json"),"w") do f 
    write(f, JSON.json(mg))
end