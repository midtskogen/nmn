# -*- coding: utf-8 -*-
"""
Data catalog for meteor showers.

This file contains the orbital and radiant data for various meteor showers,
based on information from the International Meteor Organization (IMO).
Source: http://www.imo.net/calendar/2014
"""
from typing import List

class ShowerInfo:
    """A class to hold information about a single meteor shower."""
    def __init__(self, name: str, name_sg: str, v_inf: float, ra: List[float], 
                 dec: List[float], rad_date: List[str], beg_date: str, end_date: str):
        self.name = name
        self.name_sg = name_sg
        self.v_inf = v_inf
        self.ra = ra
        self.dec = dec
        self.rad_date = rad_date
        self.beg_date = beg_date
        self.end_date = end_date

# List of all meteor showers
showerlist: List[ShowerInfo] = [
    ShowerInfo('kvadrantidene', 'kvadrantide', 41.0, [226, 228, 231, 234], [50, 50, 49, 48], ['12-30', '01-01', '01-05', '01-10'], '12-28', '01-12'),
    ShowerInfo('alpha-kentauridene', 'alpha-kentauride', 56.0, [200, 208, 214, 220, 225], [-57, -59, -60, -62, -63], ['01-30', '02-05', '02-10', '02-15', '02-20'], '01-28', '02-21'),
    ShowerInfo('gamma-normidene', 'gamma-normide', 56.0, [225, 230, 235, 240, 245], [-51, -50, -50, -50, -49], ['02-28', '03-05', '03-10', '03-15', '03-20'], '02-25', '03-22'),
    ShowerInfo('lyridene', 'lyride', 49.0, [263, 269, 274], [34, 34, 34], ['04-15', '04-20', '04-25'], '04-16', '04-25'),
    ShowerInfo('eta-puppidene', 'eta-puppide', 18.0, [106, 109, 111], [-44, -45, -45], ['04-15', '04-20', '04-25'], '04-15', '04-28'),
    ShowerInfo('eta-akvaridene', 'eta-akvaride', 66.0, [323, 328, 332, 337, 341, 345, 349, 353], [-7, -5, -3, -1, 1, 3, 5, 7], ['04-20', '04-25', '04-30', '05-05', '05-10', '05-15', '05-20', '05-25'], '04-19', '05-28'),
    ShowerInfo('eta-lyridene', 'eta-lyride', 43.0, [283, 288, 293], [44, 44, 45], ['05-05', '05-10', '05-15'], '05-03', '05-14'),
    ShowerInfo('juni-bootidene', 'juni-bootide', 18.0, [223, 224, 225], [48, 47.5, 47], ['06-25', '06-27', '06-30'], '06-22', '07-02'),
    ShowerInfo('piscis-austrinidene', 'piscis-austrinide', 35.0, [330, 334, 338, 343, 348, 352], [-34, -33, -31, -29, -27, -26], ['07-15', '07-20', '07-25', '07-30', '08-05', '08-10'], '07-15', '08-10'),
    ShowerInfo('sørlige delta-akvaridene', 'sørlig delta-akvaride', 41.0, [325, 329, 333, 337, 340, 345, 349, 352, 356], [-19, -19, -18, -17, -16, -14, -13, -12, -11], ['07-10', '07-15', '07-20', '07-25', '07-30', '08-05', '08-10', '08-15', '08-20'], '07-12', '08-23'),
    ShowerInfo('alpha-capricornidene', 'alpha-capricornide', 23.0, [285, 289, 294, 299, 303, 307, 313, 318], [-16, -15, -14, -12, -11, -10, -8, -6], ['07-05', '07-10', '07-15', '07-20', '07-25', '07-30', '08-05', '08-10'], '07-03', '08-15'),
    ShowerInfo('perseidene', 'perseide', 59.0, [6, 11, 22, 29, 37, 45, 51, 57, 63], [50, 52, 53, 54, 56, 57, 58, 58, 58], ['07-15', '07-20', '07-25', '07-30', '08-05', '08-10', '08-15', '08-20', '08-25'], '07-17', '08-24'),
    ShowerInfo('kappa-cygnidene', 'kappa-cygnide', 25.0, [283, 284, 285, 286, 288, 289], [58, 58, 59, 59, 60, 60], ['08-05', '08-10', '08-15', '08-20', '08-25', '08-30'], '08-03', '08-25'),
    ShowerInfo('alpha-aurigidene', 'alpha-aurigide', 66.0, [85, 90, 96, 102], [40, 39, 39, 38], ['08-25', '08-30', '09-05', '09-10'], '08-25', '09-05'),
    ShowerInfo('september-epsilon-perseidene', 'september-epsilon-perseide', 64.0, [43, 48, 53, 59], [40, 40, 40, 41], ['09-05', '09-10', '09-15', '09-20'], '09-05', '09-21'),
    ShowerInfo('draconidene', 'draconide', 20.0, [262, 262, 262], [64, 64, 64], ['10-05', '10-10', '10-15'], '10-06', '10-10'),
    ShowerInfo('sørlige tauridene', 'sørlig tauride', 27.0, [12, 15, 18, 21, 25, 28, 32, 36, 40, 43, 47, 52, 56, 60, 64], [3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 15, 16, 16], ['09-10', '09-15', '09-20', '09-25', '09-30', '10-05', '10-10', '10-15', '10-20', '10-25', '10-30', '11-05', '11-10', '11-15', '11-20'], '09-10', '11-20'),
    ShowerInfo('delta-aurigidene', 'delta-aurigide', 64.0, [82, 87, 92], [45, 43, 41], ['10-10', '10-15', '10-20'], '10-10', '10-18'),
    ShowerInfo('epsilon-geminidene', 'epsilon-geminide', 70.0, [99, 104, 109], [27, 27, 27], ['10-15', '10-20', '10-25'], '10-14', '10-27'),
    ShowerInfo('orionidene', 'orionide', 66.0, [85, 88, 91, 94, 98, 101, 105], [14, 15, 15, 16, 16, 16, 17], ['10-05', '10-10', '10-15', '10-20', '10-25', '10-30', '11-05'], '10-02', '11-07'),
    ShowerInfo('leo-minoridene', 'leo-minoride', 62.0, [158, 163, 168], [39, 37, 35], ['10-20', '10-25', '10-30'], '10-19', '10-27'),
    ShowerInfo('nordlige tauridene', 'nordlig tauride', 29.0, [38, 43, 47, 52, 56, 61, 65, 70, 74], [18, 19, 20, 21, 22, 23, 24, 24, 24], ['10-20', '10-25', '10-30', '11-05', '11-10', '11-15', '11-20', '11-25', '11-30'], '10-20', '12-10'),
    ShowerInfo('leonidene', 'leonide', 71.0, [147, 150, 153, 156, 159], [24, 23, 21, 20, 19], ['11-10', '11-15', '11-20', '11-25', '11-30'], '11-06', '11-30'),
    ShowerInfo('alpha-monocerotidene', 'alpha-monocerotide', 65.0, [112, 116, 120], [2, 1, 0], ['11-15', '11-20', '11-25'], '11-15', '11-25'),
    ShowerInfo('phoenicidene', 'phoenicide', 18.0, [14, 18, 22], [-52, -53, -53], ['11-30', '12-05', '12-10'], '11-28', '12-09'),
    ShowerInfo('puppidene/velidene', 'puppide/velide', 40.0, [120, 122, 125, 128], [-45, -45, -45, -45], ['11-30', '12-05', '12-10', '12-15'], '12-01', '12-15'),
    ShowerInfo('monocerotidene', 'monocerotide', 42.0, [100, 102, 104], [8, 8, 8], ['12-10', '12-12', '12-15'], '11-27', '12-17'),
    ShowerInfo('sigma-hydridene', 'sigma-hydride', 58.0, [122, 126, 130], [3, 2, 1], ['12-05', '12-10', '12-15'], '12-03', '12-15'),
    ShowerInfo('geminidene', 'geminide', 35.0, [103, 108, 113, 118], [33, 33, 33, 32], ['12-05', '12-10', '12-15', '12-20'], '12-04', '12-17'),
    ShowerInfo('comae-berencidene', 'comae-berencide', 65.0, [101, 177, 180], [19, 18, 16], ['12-15', '12-20', '12-25'], '12-12', '12-23'),
    ShowerInfo('desember-leonis-minoridene', 'desember-leonis-minoride', 64.0, [149, 153, 157, 161, 166, 170, 172, 176, 180, 185, 189, 193, 198, 203], [37, 35, 33, 31, 28, 26, 25, 23, 21, 19, 17, 15, 12, 10], ['12-05', '12-10', '12-15', '12-20', '12-25', '12-30', '01-01', '01-05', '01-10', '01-15', '01-20', '01-25', '01-30', '02-05'], '12-05', '02-04'),
    ShowerInfo('ursidene', 'urside', 33.0, [217, 217, 217], [76, 75, 74], ['12-20', '12-22', '12-25'], '12-17', '12-26'),
    ShowerInfo('mai-camelopardalidene', 'mai-camelopardalide', 18.0, [124, 124, 124], [79, 79, 79], ['05-10', '05-20', '05-30'], '05-10', '05-31')
]
