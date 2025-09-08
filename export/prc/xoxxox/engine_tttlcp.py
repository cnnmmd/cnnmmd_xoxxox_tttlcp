import re
from llama_cpp import Llama
from xoxxox.shared import Custom, LibLog

#---------------------------------------------------------------------------

class TttPrc:
  def __init__(self, config="xoxxox/config_tttlcp_cmm001", **dicprm):
    diccnf = Custom.update(config, dicprm)
    self.pthmdl = "/opt/applcp/prm"
    self.mdlold = ""
    self.conlog = {}

  def status(self, config="xoxxox/config_tttlcp_cmm001", **dicprm):
    diccnf = Custom.update(config, dicprm)
    mdlcrr = diccnf["nmodel"]
    numtrd = diccnf["numtrd"]
    numctx = diccnf["numctx"]
    numgpu = diccnf["numgpu"]
    if mdlcrr != self.mdlold:
      self.objmdl = Llama(
        model_path=f"{self.pthmdl}/{mdlcrr}",
        n_ctx=numctx,
        n_threads=numtrd,
        n_gpu_layers=numgpu,
        verbose=False
      )
      self.mdlold = mdlcrr
    self.numtmp = diccnf["numtmp"]
    self.numtop = diccnf["numtop"]
    self.maxtkn = diccnf["maxtkn"]
    self.expert = diccnf["expert"]
    if not (self.expert in self.conlog):
      self.conlog[self.expert] = LibLog.getlog(diccnf["conlog"]) # LOG
      self.conlog[self.expert].catsys(diccnf) # LOG

  def infere(self, txtreq):
    prompt = self.conlog[self.expert].catreq(txtreq) # LOG
    print("prompt[", prompt, "]", sep="", flush=True) # DBG
    rawifr = self.objmdl(
      prompt,
      max_tokens=self.maxtkn,
      temperature=self.numtmp,
      top_p=self.numtop,
      stop=None,
      echo=False
    )
    print("rawifr[", rawifr, "]", sep="", flush=True) # DBG
    txtifr = ""
    if "choices" in rawifr and len(rawifr["choices"]) > 0:
      txtifr = rawifr["choices"][0].get("text", "")
    elif "text" in rawifr:
      txtifr = rawifr["text"]
    txtifr = txtifr.strip()
    print("txtifr[" + txtifr + "]", flush=True) # DBG
    txtres, txtopt = self.conlog[self.expert].arrres(txtifr) # LOG
    print("txtres[" + txtres + "]", flush=True) # DBG
    print("txtopt[" + txtopt + "]", flush=True) # DBG
    self.conlog[self.expert].catres(txtres) # LOG
    return (txtres, txtopt)
