#!/usr/bin/env python3
import base64, hashlib, hmac, json, os, struct, subprocess, sys, tempfile, time

helper=sys.argv[1]
secret=bytes(range(32)); generation=0x0102030405060708
timeline_revision=1

def add(value: bytes): return struct.pack(">Q",len(value))+value
def u64(value): return struct.pack(">Q",value)
def fingerprint(kind,source):
    body=b"\x01"+add(b"DonutStudio/ProgrammablePayload")+add(kind.encode())+add(source.encode())
    return hashlib.sha256(body).hexdigest()
def grant(kind,source,nonce,gen=None,mac_kind=None,issued=None):
    gen=generation if gen is None else gen; now=int(time.time()*1000) if issued is None else issued
    values=[3,gen,nonce,now,1,0,0,0,25,50,64]
    fp=fingerprint(kind,source)
    body=b"\x03"+add(b"DonutStudio/RuntimeGrant/HMAC-SHA256")+u64(values[0])+add(kind.encode())+add(fp.encode())+add(b"")+add(b"")
    body+=b"".join(u64(v) for v in values[1:])+add((mac_kind or kind).encode())
    mac=hmac.new(secret,body,hashlib.sha256).hexdigest()
    return {"version":3,"kind":kind,"fingerprint":fp,"catalogPackId":"","catalogProgramId":"",
            "sessionGeneration":gen,"nonce":nonce,"issuedAtMs":now,"approved":True,"disk":False,
            "network":False,"verifiedBundledCurated":False,"cpuMs":25,"gpuMs":50,"memoryMiB":64,"mac":mac}

def params_for(method,source,g):
    if method=="viewport_set_script": return {"source":source,"lang":"lua",**({} if g is None else {"runtimeGrant":g})}
    if method=="shader_compile": return {"source":source,**({} if g is None else {"runtimeGrant":g})}
    if method=="script_compile": return {"source":source,"lang":"lua",**({} if g is None else {"runtimeGrant":g})}
    result={"scriptLang":"lua","luaScript":source,"width":16,"height":16,"fps":1,"durationSec":0.01}
    if method=="viewport_set_timeline": result["authoringRevision"]=timeline_revision
    if g is not None: result["scriptRuntimeGrant"]=g
    return result

def message(response):
    value=response.get("error",{})
    return value.get("message",str(value)) if isinstance(value,dict) else str(value)

read_fd,write_fd=os.pipe()
try: saved_fd3=os.dup(3)
except OSError: saved_fd3=None
if read_fd != 3:
    os.dup2(read_fd,3); os.close(read_fd)
os.write(write_fd,secret+generation.to_bytes(8,"big")); os.close(write_fd)
with tempfile.TemporaryDirectory() as matte, tempfile.TemporaryDirectory() as depth:
    process=subprocess.Popen([helper,"--matte-cache-root",matte,"--depth-cache-root",depth],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,pass_fds=(3,),text=True,bufsize=1)
    if saved_fd3 is None: os.close(3)
    else: os.dup2(saved_fd3,3); os.close(saved_fd3)
    methods=["viewport_set_script","shader_compile","script_compile","viewport_set_timeline",
             "export","render_cache_build","composite_frame_probe"]
    request_id=0; nonce=10
    def send(method,params):
        nonlocal_dummy=None
        global request_id
        request_id+=1
        process.stdin.write(json.dumps({"id":request_id,"method":method,"params":params})+"\n"); process.stdin.flush()
        while True:
            line=process.stdout.readline()
            if not line: raise AssertionError(f"helper exited {process.poll()}: {process.stderr.read()}")
            response=json.loads(line)
            if response.get("id")==request_id: return response
    opened=send("viewport_open",{"width":16,"height":16})
    assert opened.get("result",{}).get("ok") is True, opened
    for method in methods:
        source="void main(){}" if method=="shader_compile" else "function frame(ctx) return 1 end"
        expected="shader" if method=="shader_compile" else "lua"
        valid=grant(expected,source,nonce); nonce+=1
        first=send(method,params_for(method,source,valid))
        if method == "viewport_set_script":
            assert first.get("result") == {"ok": True}, (method,first)
        elif method == "shader_compile":
            # Authored shaders are intentionally parked until a sandbox exists;
            # an authenticated non-curated grant must fail closed at that exact gate.
            assert message(first) == "sandbox unavailable", (method,first)
        elif method == "script_compile":
            result=first.get("result",{})
            assert result == {"ok":True,"engine":"lua","available":True,"error":""}, (method,first)
        elif method == "viewport_set_timeline":
            result=first.get("result",{})
            assert result.get("ok") is True, (method,first)
            assert result.get("authoringRevision") == result.get("acceptedRevision") == timeline_revision, (method,first)
            assert result.get("compiledRevision") == result.get("lastGoodRevision") == timeline_revision, (method,first)
        else:
            # The minimal job/snapshot fixture asserts the real bounded route reached
            # its next production validation boundary, never a generic admission pass.
            assert "error" in first, (method,first)
            text=message(first)
            assert "authentication failed" not in text and "runtimeGrant" not in text, (method,first)
            assert any(token in text.lower() for token in ("snapshot", "segment", "output", "path", "source", "viewport", "timeline")), (method,first)
        replay=send(method,params_for(method,source,valid))
        assert "replay" in message(replay), (method,replay)
        invalids=[None,grant(expected,source,nonce,generation-1),grant("javascript",source,nonce+1),
                  grant(expected,source,nonce+2,issued=int(time.time()*1000)-31000)]
        forged=grant(expected,source,nonce+3); forged["mac"]="00"*32; invalids.append(forged); nonce+=4
        for value in invalids:
            response=send(method,params_for(method,source,value))
            assert "error" in response, (method,value,response)

    # Exercise export, render-cache, and probe owners with a real one-frame media snapshot.
    media=os.path.join(matte,"tiny.mp4")
    subprocess.run(["ffmpeg","-loglevel","error","-f","lavfi","-i","color=c=red:s=16x16:r=1",
                    "-frames:v","1","-pix_fmt","yuv420p",media],check=True,timeout=10)
    plan={"clipId":1,"structuralRevision":1,"identityMode":"authoredGraph","valid":True,
          "nodeKinds":["video.legacy.source","video.legacy.retime","video.legacy.transform",
                       "video.legacy.effects","video.out"]}
    common={"segments":[{"sourcePath":media,"sourceKind":"media","clipId":1,"inSec":0.0,
                         "outSec":1.0,"displayStartSec":0.0}],
            "visualLayerPlans":[plan],"clips":[{"clipId":1}],"authoringRevision":1,
            "exportableRevision":1,"width":16,"height":16,"fps":1,"durationSec":1.0,
            "encoder":"software","scriptLang":"lua","luaScript":"function frame(ctx) return 1 end"}
    for method,name in (("export","export.mp4"),("render_cache_build","cache.mp4")):
        source=common["luaScript"]; admitted=grant("lua",source,nonce); nonce+=1
        out=os.path.join(matte,name)
        response=send(method,{**common,"outPath":out,"scriptRuntimeGrant":admitted})
        result=response.get("result",{})
        assert result.get("outPath")==out and os.path.isfile(out) and os.path.getsize(out)>0, response
        assert result.get("encoder") not in (None,""), response
        if method=="render_cache_build":
            assert result.get("frames")==1 and os.path.getsize(out)<1024*1024, response
        else:
            assert isinstance(result.get("glCompositing"),bool), response
            assert result.get("interpolationBackend") not in (None,""), response
        assert result.get("testScriptLimitsAtOperation")=={"cpuMs":25,"memoryMiB":64}, response
        assert "replay" in message(send(method,{**common,"outPath":out+".replay","scriptRuntimeGrant":admitted}))
    source=common["luaScript"]; admitted=grant("lua",source,nonce); nonce+=1
    probe_params={**common,"scriptRuntimeGrant":admitted,"timelineSec":0.0,"snapshotGeneration":9,
        "clipRevisions":[{"clipId":1,"authoredStructuralRevision":1,"identityMode":"authoredGraph",
                          "snapshotCompiledExportable":True}]}
    probe=send("composite_frame_probe",probe_params); result=probe.get("result",{})
    pixels=base64.b64decode(result.get("rgbaBase64",""),validate=True)
    assert result.get("width")==16 and result.get("height")==16 and result.get("strideBytes")==64, probe
    assert result.get("format")=="rgba8" and len(pixels)==16*16*4, probe
    assert result.get("snapshotGeneration")==9 and result.get("graphLayerCount")==1, probe
    assert result.get("compositorBackend") not in (None,"") and result.get("presentationBackend") not in (None,""), probe
    assert result.get("testScriptLimitsAtOperation")=={"cpuMs":25,"memoryMiB":64}, probe
    assert "replay" in message(send("composite_frame_probe",probe_params))

    malformed=[
        {"segments":{}}, {"clips":"bad"}, {"segments":[7]}, {"clips":[False]},
        {"scriptLang":"lua","jsScript":"function frame(){}"},
        {"scriptLang":"js","luaScript":"function frame() return 1 end"},
        {"segments":[{"shaderSource":"void main(){}"}]},
    ]
    for method in ("export","render_cache_build","composite_frame_probe"):
        for index,bad in enumerate(malformed):
            out=os.path.join(matte,f"malformed-{method}-{index}.mp4")
            bad={**bad,"outPath":out}
            response=send(method,bad)
            assert "error" in response and 0 < len(message(response)) < 512, (method,bad,response)
            assert not os.path.exists(out) and not os.path.exists(out+".part"), out
            assert send("ping",{}).get("result") == "pong"
    request_id+=1
    process.stdin.write(json.dumps({"id":request_id,"method":"shutdown","params":{}})+"\n"); process.stdin.flush()
    process.wait(timeout=10); assert process.returncode==0, process.stderr.read()
print("raw programmable RPC valid/replay/rejection matrix passed")
