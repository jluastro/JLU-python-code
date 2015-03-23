from astropy.table import Table 



def match_named(label, starl_name):
    '''
    read in both files, then find first 2 named stars that are in label file
    then rewrite starlist file with those two stars as the first stars 
    '''

    lab = Table.read(label, format='ascii.no_header')
    nameL = lab['col1']

    starlist = Table.read(starl_name, format='ascii.no_header')
    nameS = starlist['col1']

   
    
    
    i = 0
    match = 0
    ml = []
    while match < 2:
        
        for ii in range(len(nameS)):
            if nameS[ii] == nameL[i]:
                match +=1
                ml.append(ii)

        i+=1

    f = open(starl_name.replace('.lis', '_rorg.lis'), 'w')

    f.write('%13s  %6.3f  %8.3f  %10.5f  %10.5f  ' %
            (starlist['col1'][ml[0]], starlist['col2'][ml[0]], starlist['col3'][ml[0]],starlist['col4'][ml[0]], starlist['col5'][ml[0]]))
    f.write('\n')

    f.write('%13s  %6.3f  %8.3f  %10.5f  %10.5f  ' %
            (starlist['col1'][ml[1]], starlist['col2'][ml[1]], starlist['col3'][ml[1]],starlist['col4'][ml[1]], starlist['col5'][ml[1]]))
    f.write('\n')
    
    for i in range(len(nameL)):
        if not i in  ml:
            f.write('%13s  %6.3f  %8.3f  %10.5f  %10.5f  ' %
                    (starlist['col1'][i], starlist['col2'][i], starlist['col3'][i],starlist['col4'][i], starlist['col5'][i]))
            f.write('\n')

    f.close()


def mat_all(starlist_list, label):

    for i in starlist_list:
        match_named(i,label)

    
        
