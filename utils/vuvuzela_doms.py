#!/usr/bin/env python


# produce  dictionary for each DOM tested. mark up the number
# of the good runs that shall represent a DOM's performance
# at a given temperature setting.
#
# That info is taken out of results_FINAL.ods
#------------------------------------------------------------------


A = {'name':'Chabahar',
     'T':18,
     'inice':'03-40',
     '0.25pe':0.25,
     'scale':150,
     'spe':1.00,
     'qpair':20
}


B = {'name':'Cloudsat',
     'T':13,
     'inice':'13-51',
     '0.25pe':0.25,
     'scale':150,
     'spe':1.00,
     'qpair':20
}

C = {'name':'Durk',
     'T':29,
     'inice':'20-11',
     '0.25pe':0.25,
     'scale':150,
     'spe':1.00,
     'qpair':20
}

D = {'name':'St_Anton',
     'T':10,
     'inice':'22-58',
     '0.25pe':0.25,
     'scale':150,
     'spe':1.00,
     'qpair':20
}

E = {'name':'Columbia',
     'T':25,
     'inice':"40-22",
     '0.25pe':0.25,
     'scale':150,
     'spe':1.00,
     'qpair':20
}


F = {'name':'Ekedalsgatan',
     'T':21,
     'inice':'77-33',
     '0.25pe':0.25,
     'scale':150,
     'spe':1.00,
     'qpair':20
}


G = {'name':'Flodtagging',
     'T':15,
     'inice':'81-25',
     '0.25pe':0.25,
     'scale':150,
     'spe':1.00,
     'qpair':20
}


H = {'name':'Hallon',
     'T':10,
     'inice':'82-51',
     '0.25pe':0.25,
     'scale':150,
     'spe':1.00,
     'qpair':20
}

I = {'name':'Mark',
     'T':15,
     'inice':'85-33',
     '0.25pe':0.25,
     'scale':150,
     'spe':1.00,
     'qpair':20
}




doms_to_plot = [A,B,C,D,E,F,G,H,I]
