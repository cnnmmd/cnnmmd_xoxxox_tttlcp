import re
from llama_cpp import Llama
from xoxxox.shared import Custom, LibLog

#---------------------------------------------------------------------------

class TttPrc:
  def __init__(self, config="xoxxox/config_tttlcp_000", **dicprm):
    diccnf = Custom.update(config, dicprm)
    self.pthmdl = "/opt/applcp/prm"
    self.mdlold = ""

  def status(self, config="xoxxox/config_tttlcp_000", **dicprm):
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
    self.conlog = LibLog.getlog(diccnf["conlog"]) # LOG
    self.numtmp = diccnf["numtmp"]
    self.numtop = diccnf["numtop"]
    self.maxtkn = diccnf["tknmax"]
    self.conlog.catsys(diccnf) # LOG

  def infere(self, txtreq):
    prompt = self.conlog.catreq(txtreq) # LOG
    print("prompt[" + prompt + "]", flush=True) # DBG
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
    txtres, txtopt = self.conlog.arrres(txtifr) # LOG
    print("txtres[" + txtres + "]", flush=True) # DBG
    print("txtana[" + txtopt + "]", flush=True) # DBG
    self.conlog.catres(txtres) # LOG
    return (txtres, txtopt)
