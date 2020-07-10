# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 17:18:36 2020

@author: Gold
"""
dat_code=input('Need data? Enter the code 131 below\n')

if dat_code=='131':
    print('Make a selection by typing the corresponding digit of your choice')
    print('1.Data Plans')
    print("2.XtraValue(Data+voice)")
    print('3.Social Bundles')
    print('4.Balance Check')
    print("5.Roaming and Int'l Offers")
    print('6.Xtratime/Byte')
    print('7.Gift Data')
    print('8.Video Packs')
    print('9.MTN4ME')
    print('99.Next')
    codee=input('Enter your number\n')
    if codee=='1':
        print('Daily Plans')
        print('1.1.N50 for 37.5MB')
        print('2.N100 for 112.5MB')
        print('3.N200 for 200MB')
        print('4.N300 for 1GB')
        print('5.N500 for 2GB(2 days)')
        print('6.N25 for WhatsApp Daily')
        print('0.Back')
    elif codee=='2':
        print('This Plan gives you airtime for Natl and Intl calls plus data after subscribing to a bundle plan ')
        print('1.XtrsTalk')
        print('XtraData')
        print("Eligible Int's Destination")
    elif codee=='3':
        print('1.WhatsApp')
        print("2.Facebook")
        print('3.Instagram')
        print('4.2GO')
        print("5.WeChat")
        print('6.Eskimi')
        print('7.All social bundles')
        print('8.Youtube and Instagram')
        print('9.Opera Mini and News')
    elif codee=='4':
        print('Your monthly balance is 1561.15MB expires 11/07/2020')
    elif codee=='5':
        print('1.Roaming Data Bundles')
        print('2.Roaming voice + Data Bundles')
        print('3.Free international roaming call')
        print("4.Int'l Calling Bundle")
    elif codee=='6':
        print('Welcome to MTN XtraTime. You can borrow up to N1500. Service fee includes VAT')
        print('1.N1500')
        print('2.N1000')
        print('3.N750')
        print('4.N500')
        print('5.Borrow data')
        print('6.My Account')
    elif codee=='7':
        print('1.Transfer from Data Balance')
        print('2.Buy for a friend')
        print('3.Request from a Friend')
        print('4.View Pending Request')
        print('0.Back')
        print('00.Main Menu')
    elif codee=='8':
        print('Video Streaming Packs')
        print('1.Youtube Video Packs')
        print('2.StarTimes Video Packs')
        print('3.1GB (YouTube Only)+500MB(Data acess)')
    elif codee=='9':
        print('1.MTN Top Deals 4ME')
        print('2.Recharge Offers 4ME')
        print('3.Data Offers 4ME')
        print('4.COMBO Bundles 4ME')
    elif codee=='99':
        print('10.Special Welcome Back Data bundles')
        print('11.Balance Check Independence')
        print('12.10MB Daily')
        print('Blance Check 4G')
        print('14.Interactive Opt In')
        print('14.Interactive Opt Out')
        print('0.Back')
    else:
        print('You have not entered any valid code. Please enter a valid code')
        
else:
    print('You did not enter the valid code')

