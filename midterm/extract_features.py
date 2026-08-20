#!/usr/bin/env python3
"""extract_features.py (v3) - Feature extractor for hybrid malware classification."""
import argparse, csv, json, os, shutil, subprocess, sys
from collections import Counter

OS_NOISE_SUFFIXES = (
    "microsoft.com","windows.com","windowsupdate.com","msftconnecttest.com","msftncsi.com",
    "live.com","office.com","office365.com","office.net","msn.com","bing.com","msedge.net",
    "azureedge.net","windows.net","microsoftonline.com","skype.com","teams.microsoft.com",
    "digicert.com","verisign.com","entrust.net","sectigo.com","globalsign.com",
    "root-servers.net","in-addr.arpa","ip6.arpa","wpad","isatap","akadns.net",
    "akamaiedge.net","edgekey.net",
)
NET_API_HINTS = ("connect","send","recv","wsasend","wsarecv","wsaconnect","wsasocket",
    "wsastartup","socket","getaddrinfo","gethostby","dnsquery","internetconnect",
    "internetopen","internetreadfile","internetcrackurl","httpopenrequest",
    "httpsendrequest","urldownloadtofile","winhttp","bind","listen","closesocket")
NET_FEATURE_KEYS = ("num_udp","num_tcp","num_icmp","num_http","num_dns_queries",
    "unique_dst_ports","unique_dst_ips","num_malware_domains")

def load_labels(p):
    d={}
    with open(p,newline="") as f:
        for r in csv.DictReader(f):
            m=(r.get("md5") or "").strip().lower()
            if m: d[m]={"filename":(r.get("filename") or "").strip(),"family":(r.get("family") or "unknown").strip()}
    return d

def load_report(p):
    with open(p,"r",errors="replace") as f: return json.load(f)

def target_md5(rep):
    fi=(rep.get("target",{}) or {}).get("file",{}) or {}
    return (fi.get("md5") or rep.get("md5") or "").lower() or None

def as_host(x):
    if isinstance(x,dict):
        for k in ("ip","hostname","host","request","domain","name"):
            v=x.get(k)
            if isinstance(v,str) and v: return v
        return ""
    return str(x) if x is not None else ""

def is_noise(h):
    h=(h or "").lower().rstrip(".")
    return (not h) or any(h==s or h.endswith("."+s) for s in OS_NOISE_SUFFIXES)

def pick_proc(procs,fname):
    if not procs: return None
    fn=(fname or "").lower()
    for p in procs:
        pn=(p.get("process_name") or "").lower(); pp=(p.get("module_path") or "").lower()
        if fn and (fn==pn or fn in pp): return p
    b=max(procs,key=lambda p: len(p.get("calls",[]) or []))
    return b if (b.get("calls") or []) else procs[0]

def api_seq(proc): return [c.get("api") for c in (proc.get("calls",[]) or []) if c.get("api")]

def is_net_call(c):
    if (c.get("category") or "").lower()=="network": return True
    a=(c.get("api") or "").lower(); return any(h in a for h in NET_API_HINTS)

def pcap_dns(pcap):
    if not shutil.which("tshark") or not os.path.isfile(pcap): return []
    try:
        r=subprocess.run(["tshark","-r",pcap,"-Y","dns.qry.name","-T","fields","-e","dns.qry.name"],
                         capture_output=True,text=True,timeout=90)
    except Exception: return []
    names=set()
    for line in r.stdout.splitlines():
        for nm in line.replace("\t",",").split(","):
            nm=nm.strip().lower()
            if nm and not is_noise(nm): names.add(nm)
    return sorted(names)

def net_numeric(rep,domains):
    net=rep.get("network",{}) or {}
    udp=[p for p in (net.get("udp",[]) or []) if isinstance(p,dict)]
    tcp=[p for p in (net.get("tcp",[]) or []) if isinstance(p,dict)]
    dports={p.get("dport") for p in udp+tcp if p.get("dport") is not None}
    dsts={p.get("dst") for p in udp+tcp if p.get("dst")}
    dns_q=sum(1 for p in udp if p.get("dport")==53)
    return {"num_udp":len(udp),"num_tcp":len(tcp),"num_icmp":len(net.get("icmp",[]) or []),
            "num_http":len(net.get("http",[]) or []),"num_dns_queries":dns_q,
            "unique_dst_ports":len(dports),"unique_dst_ips":len(dsts),"num_malware_domains":len(domains)}

def process_one(rp,labels):
    rep=load_report(rp); m=target_md5(rep)
    if not m or m not in labels: return None
    lab=labels[m]; tid=(rep.get("info",{}) or {}).get("id")
    procs=(rep.get("behavior",{}) or {}).get("processes",[]) or []
    proc=pick_proc(procs,lab["filename"]); seq=api_seq(proc) if proc else []
    pcap=os.path.join(os.path.dirname(os.path.dirname(rp)),"dump.pcap")
    doms=pcap_dns(pcap); nf=net_numeric(rep,doms)
    return {"task_id":tid,"md5":m,"filename":lab["filename"],"family":lab["family"],
            "malware_process":(proc.get("process_name") if proc else None),"num_processes":len(procs),
            "api_sequence":seq,"api_sequence_length":len(seq),"api_counts":dict(Counter(seq)),
            "unique_apis":sorted(set(seq)),"network_features":nf,"malware_domains":doms,
            "process_network_call_count":(sum(1 for c in (proc.get("calls",[]) or []) if is_net_call(c)) if proc else 0)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--analyses",default="/opt/CAPEv2/storage/analyses")
    ap.add_argument("--labels",default="/opt/samples/labels.csv")
    ap.add_argument("--out",default="/opt/samples/features")
    a=ap.parse_args()
    if not shutil.which("tshark"): print("[!] tshark missing - domains will be empty (sudo apt install -y tshark)\n")
    labels=load_labels(a.labels)
    if not labels: print("[!] no labels"); sys.exit(1)
    print(f"[+] Loaded {len(labels)} labelled samples")
    os.makedirs(a.out,exist_ok=True); ds=[]
    for e in sorted((x for x in os.listdir(a.analyses) if x.isdigit()),key=int):
        rp=os.path.join(a.analyses,e,"reports","report.json")
        if not os.path.isfile(rp): continue
        try: fe=process_one(rp,labels)
        except Exception as ex: print(f"[!] task {e}: {ex}"); continue
        if fe is None: continue
        ds.append(fe)
        with open(os.path.join(a.out,f"{fe['family']}__{os.path.splitext(fe['filename'])[0]}.json"),"w") as fh:
            json.dump(fe,fh,indent=2)
    with open(os.path.join(a.out,"dataset.json"),"w") as fh: json.dump(ds,fh,indent=2)
    print(f"\n[+] Extracted {len(ds)} samples -> {a.out}/dataset.json")
    print(f"{'task':>4}  {'family':<12} {'api_seq':>8} {'dns_q':>6} {'domains':>7} {'uniq_ips':>8}  process")
    print("-"*78)
    for d in sorted(ds,key=lambda x:(x['task_id'] or 0)):
        nf=d["network_features"]
        print(f"{str(d['task_id']):>4}  {d['family']:<12} {d['api_sequence_length']:>8} {nf['num_dns_queries']:>6} {nf['num_malware_domains']:>7} {nf['unique_dst_ips']:>8}  {d['malware_process']}")
    print("-"*78); print("class balance:",dict(Counter(d["family"] for d in ds)))
    thin=[d["filename"] for d in ds if d["api_sequence_length"]<50]
    if thin: print(f"[!] thin runs (<50 calls): {thin}")

if __name__=="__main__": main()
