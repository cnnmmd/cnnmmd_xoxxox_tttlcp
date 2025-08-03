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

  def status(self, config="xoxxox/config_tttlcp_000", **dicprm):
    diccnf = Custom.update(config, dicprm)
    # 設定：全般
    self.lsthed = []
    self.lstbdy = []
    self.frmsys = "{elmsys}\n"
    self.frmusr = "{txtsrc}{txtdef}{elmusr}\n"
    self.frmagt = "{txtdst}{txtdef}{elmagt}\n"
    # 設定：個別
    self.numtmp = diccnf["numtmp"]
    self.numtop = diccnf["numtop"]
    self.maxtkn = diccnf["tknmax"]
    self.maxbdy = diccnf["prmmax"]
    self.nulagt = diccnf["nuloth"]
    self.dictlk = {
      "elmsys": diccnf["status"],
      "txtsrc": diccnf["rolslf"],
      "txtdst": diccnf["roloth"],
      "elmusr": diccnf["inislf"],
      "elmagt": diccnf["inioth"],
      "txtdef": "＞",
    }
    self.lsthed.append(self.frmsys.format_map(self.dictlk))
    self.lstbdy.append(self.frmusr.format_map(self.dictlk))
    self.lstbdy.append(self.frmagt.format_map(self.dictlk))

  def infere(self, txtreq):
    # 生成：導入
    self.dictlk["elmusr"] = txtreq
    self.lstbdy.append(self.frmusr.format_map(self.dictlk))
    strlog = "".join(self.lsthed + self.lstbdy)
    #print("strlog[" + strlog + "]", flush=True) # DBG
    # 生成：推定
    prompt = strlog
    infenc = self.objmdl(
      prompt,
      max_tokens=self.maxtkn,
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
    #print("infdec[" + infdec + "]", flush=True) # DBG
    # 結果：加工
    try:
      elmagt = re.findall(self.dictlk["txtdst"] + self.dictlk["txtdef"] + "(.*)", infdec)[0]
    except Exception as e:
      elmagt = ""
    if elmagt == "":
      elmagt = self.nulagt
    # ログ：追加
    self.dictlk["elmagt"] = elmagt
    self.lstbdy.append(self.frmagt.format_map(self.dictlk))
    if len(self.lstbdy) > self.maxbdy * 2:
      self.lstbdy.pop(0)
      self.lstbdy.pop(0)
    # 結果：返却
    txtres = elmagt
    return txtres
