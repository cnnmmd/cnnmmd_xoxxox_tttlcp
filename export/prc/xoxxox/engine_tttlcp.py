import re
from llama_cpp import Llama
from xoxxox.shared import Custom

#---------------------------------------------------------------------------

class TttPrc:
  def __init__(self, config="xoxxox/config_tttlcp_000", **dicprm):
    diccnf = Custom.update(config, dicprm)
    pthmdl = "/opt/applcp/models"
    nmodel = diccnf["nmodel"]
    numtrd = diccnf["numtrd"]
    self.objmdl = Llama(
      model_path=f"{pthmdl}/{nmodel}",
      n_threads=numtrd,
      verbose=False
    )

  def status(self, config="xoxxox/config_tttlc_000", **dicprm):
    diccnf = Custom.update(config, dicprm)
    # 設定：全般
    self.numtmp = diccnf["numtmp"]
    self.numtop = diccnf["numtop"]
    self.tknmax = diccnf["tknmax"]
    self.lsthed = []
    self.lstbdy = []
    self.txtdef = "＞"
    self.frmsys = "f\"{elmsys}\\n\""
    self.frmusr = "f\"{self.txtsrc + self.txtdef + elmusr}\\n\""
    self.frmagt = "f\"{self.txtdst + self.txtdef + elmagt}\\n\""
    # 設定：個別
    self.maxbdy = diccnf["prmmax"]
    self.txtsrc = diccnf["rolslf"]
    self.txtdst = diccnf["roloth"]
    self.nuloth = diccnf["nuloth"]
    elmsys = diccnf["status"]
    self.lsthed.append(eval(self.frmsys))
    elmusr = diccnf["inislf"]
    self.lstbdy.append(eval(self.frmusr))
    elmagt = diccnf["inioth"]
    self.lstbdy.append(eval(self.frmagt))

  def infere(self, txtreq):
    # 生成：導入
    elmusr = txtreq
    self.lstbdy.append(eval(self.frmusr))
    strlog = "".join(self.lsthed + self.lstbdy)
    #print("strlog[" + strlog + "]") # DBG
    # 生成：推定
    prompt = strlog
    infenc = self.objmdl(
      prompt,
      max_tokens=self.tknmax,
      temperature=self.numtmp,
      top_p=self.numtop,
      stop=None,
      echo=False
    )
    infdec = ""
    if "choices" in infenc and len(infenc["choices"]) > 0:
      infdec = infenc["choices"][0].get("text", "")
    elif "text" in infenc:
      infdec = infenc["text"]
    infdec.strip()
    #print("infdec[" + infdec + "]") # DBG
    # 結果：加工
    try:
      elmagt = re.findall(self.txtdst + self.txtdef + "(.*)", infdec)[0]
    except Exception as e:
      elmagt = ""
    if elmagt == "":
      elmagt = self.nuloth
    # ログ：追加
    self.lstbdy.append(eval(self.frmagt))
    if len(self.lstbdy) > self.maxbdy * 2:
      self.lstbdy.pop(0)
      self.lstbdy.pop(0)
    # 結果：返却
    txtres = elmagt
    return txtres
