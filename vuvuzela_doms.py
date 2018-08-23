#!/usr/bin/env python


# produce  dictionary for each DOM tested. mark up the number
# of the good runs that shall represent a DOM's performance
# at a given temperature setting.
#
# That info is taken out of results_FINAL.ods
#------------------------------------------------------------------


A = {'name':'Chabahar (03-40)',
     'T':18,
     'inice':'03-40',
     '0.25pe':0.25,
     'scale':150,
     'spe':1.00,
     'qpair':20
}


B = {'name':'Cloudsat (13-51)',
     'T':13,
     'inice':'13-51',
     '0.25pe':0.25,
     'scale':150,
     'spe':1.00,
     'qpair':20
}

C = {'name':'Durk (20-11)',
     'T':29,
     'inice':'20-11',
     '0.25pe':0.25,
     'scale':150,
     'spe':1.00,
     'qpair':20
}

D = {'name':'St_Anton (22-58)',
     'T':10,
     'inice':'22-58',
     '0.25pe':0.25,
     'scale':150,
     'spe':1.00,
     'qpair':20
}

E = {'name':'Columbia (40-22)',
     'T':25,
     '0.25pe':0.25,
     'scale':150,
     'spe':1.00,
     'qpair':20
}


F = {'name':'Ekedalsgatan (77-33)',
     'T':21,
     'inice':'77-33',
     '0.25pe':0.25,
     'scale':150,
     'spe':1.00,
     'qpair':20
}


G = {'name':'Flodtagging (81-25)',
     'T':15,
     'inice':'81-25',
     '0.25pe':0.25,
     'scale':150,
     'spe':1.00,
     'qpair':20
}


H = {'name':'Hallon (82-51)',
     'T':10,
     'inice':'82-51',
     '0.25pe':0.25,
     'scale':150,
     'spe':1.00,
     'qpair':20
}

I = {'name':'Mark (85-33)',
     'T':15,
     'inice':'85-33',
     '0.25pe':0.25,
     'scale':150,
     'spe':1.00,
     'qpair':20
}




doms_to_plot = [A,B,C,D,E,F,G,H,I]
