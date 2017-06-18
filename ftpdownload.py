# This Python file uses the following encoding: utf-8
# ftp download data
from ftplib import FTP
import datetime

def ftpconnect():
	ftp_server = '198.118.195.100'
	username = 'sijia.zhang1@qq.com'
	password = 'sijia.zhang1@qq.com'
	ftp = FTP()
	ftp.set_debuglevel(2) #打开调试级别2，显示详细信息
	ftp.connect(ftp_server,21) #连接
	ftp.login(username,password) #登录，如果匿名登录则用空串代替即可
	#ftp.set_pasv(Flase)
	return ftp

def downloadfile():  
	ftp = ftpconnect()    
	print ftp.getwelcome() #显示ftp服务器欢迎信息
	print 'IN______________________________'
	datapath1 = "/gpmdata/"	 #/trmmdata/ByDate/V07
	datapath2 = "/radar/"

	# 日期-路径
	dbegin = datetime.datetime(2017,5,1) #下载数据的开始日期
	dend = datetime.datetime(2017,5,10)  #下载数据的结束日期
	d = dbegin
	delta = datetime.timedelta(days=1)
	while d <= dend:
		day   = d.strftime("%d")
		month = d.strftime("%m")
		year  = d.strftime("%Y")
		path  = datapath1+year+'/'+month+'/'+day+datapath2  #下载路径/gpmdata/2014/08/01/imerg/
		#---------------------
		#下载路径下的所有文件
		#li = ftp.nlst(path) #获得path路径下的目录列表
		#下载路径下的部分文件
		li = ftp.nlst(path) 
		li2=[]
		for line in li:
			if line.startswith(path+'2A.GPM.DPR.V7'):
				li2.append(line)
				print line
		#---------------------
		for eachFile in li2:
			localpaths = eachFile.split("/")
			print len(localpaths)
			localpath = localpaths[len(localpaths)-1]#u'E:/GPM-IMERG/GPM DPR/'
			localpath = u'F:/'+localpath #把日期放在最前面，方便排序
			#localpath='G:\GPM-IMERG/3imerghhr/'+year+month+'/'+year+month+day+'/'+localpath #把日期放在最前面，方便排序
			bufsize = 1024 #设置缓冲块大小      
			fp = open(localpath,'wb') #以写模式在本地打开文件
			ftp.retrbinary('RETR ' + eachFile,fp.write,bufsize) #接收服务器上文件并写入本地文件
		d += delta 

	ftp.set_debuglevel(0) #关闭调试
	fp.close()
	ftp.quit() #退出ftp服务器

if __name__=="__main__":
	downloadfile()

