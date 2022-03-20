
import glob



folderlist=sorted(glob.glob("/zfs/helios/filer1/idayan/CALed/202010*")) \
+ sorted(glob.glob("/zfs/helios/filer1/idayan/CALed/202011*")) \
+ sorted(glob.glob("/zfs/helios/filer1/idayan/CALed/202102*")) \
+ sorted(glob.glob("/zfs/helios/filer1/idayan/CALed/202104*"))



#for j in folderlist:
#    globbed=glob.glob("{}/*.fits".format(j))
#    print("printing folder")
#    print(j)
for j in folderlist:
    globbed=sorted(glob.glob("{}/*.fits".format(j)))
    print("printing folder")
    print(j)
    #print((globbed))
    #with open(r'/home/idayan/imgsin60-2-10110204.txt', 'a') as f:
    with open(r'/home/idayan/TESTimgsin604.txt', 'a') as f:
                 f.write('\n'.join(globbed))
    

    
  

  
