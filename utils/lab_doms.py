#!/usr/bin/env python


# produce  dictionary for each DOM tested. mark up the number
# of the good runs that shall represent a DOM's performance
# at a given temperature setting.
#
# That info is taken out of results_FINAL.ods
#------------------------------------------------------------------

Aludsparken = {'name':'Aluddsparken',
               'T10':184,
               'T20':180,
               'T30':177,
               'T40':172,
               'scale':150,
               'spe':4.104,#5.13,#m10 run must be divided by 0.8
               'qpair':30
               }

Skogshare = {'name':'Skogshare',
             'T10':136,
             'T20':129,
             'T30':124,
             'T40':143,
             'scale':150,
             'spe':0.81,
             'qpair':14
             }

Ingladsil = {'name':'Ingladsil',
             'T10':108,
             'T20':127,
             'T30':126, #,116]
             'T40':116,
             'scale':50,
             'spe':2.1,
             'qpair':7.
             }

Antarctica = {'name':'Antarctica',
              'T10':134,
              'T20':128,
              'T30':123,
              'T40':141,
              'scale': 100,
              'spe':2.12,
              'qpair':7.
               }

doms_to_plot = [Aludsparken, Skogshare, Ingladsil, Antarctica]

