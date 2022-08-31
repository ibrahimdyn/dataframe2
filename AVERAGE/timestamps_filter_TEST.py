import sys

pathsofimages=sys.argv[1]

print "PRINTING IMAGE PATHS"
print pathsofimages

print "SECOND PHASE"
with open(pathsofimages, 'r') as ff:
  lines=ff.read().splitlines()
imagepaths = sorted(lines) 
print imagepaths

#with open('/home/idayan/TOtimestamp202005121735.txt','r') as f:  #81531 from UCALED 
  
  
#with open('/home/idayan/toavrg202005181400.txt','r') as f: # 65805
#with open('/home/idayan/_Toavrg202010030948.txt','r') as f: # 31895
#with open('/home/idayan/toavrg202010030948.txt','r') as f: # 44395 #in aAVErAGED-FINAL somehor half of this date is missing after 10:12
#with open('/home/idayan/toavrg202006051431.txt','r') as f: # 71243
#with open('/home/idayan/toavrg202005121735.txt','r') as f: # wc -l  toavrg202005121735.txt 148906
#with open('/home/idayan/rmngtoavrg202009290730.txt','r') as f: # wc -l rmngtoavrg202009290730.txt /home/idayan/tbavrgd202009290730.txt 102394
#with open('/home/idayan/tbavrgd202009290730.txt','r') as f: # /home/idayan/tbavrgd202009290730.txt 218459
#with open('/home/idayan/imgs202006051431toavrg.txt','r') as f: # two sec seperated imgs; check if they really like that; 87577
#with open('/home/idayan/2-imgsin70-101102.txt','r') as f:
#with open('/home/idayan/imgsin70.txt','r') as f:

#    lines=f.read().splitlines()
#imagepaths = sorted(lines) 
