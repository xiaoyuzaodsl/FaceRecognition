global h_axes1;  
global h_axes2;  
global edit2;  
h_f=figure('name','����ʶ��ϵͳ','position',[300,200,600,400]);  
clf reset;  

set(h_f, 'defaultuicontrolfontsize' ,12);    
set(h_f, 'defaultuicontrolfontname' , '����' );    
  
h_axes1=axes('parent',h_f,'position',[0.2 0.28 0.25 0.56],'Unit','normalized','visible','off');  
h_axes2=axes('parent',h_f,'position',[0.55 0.28 0.25 0.56],'Unit','normalized','visible','off');  
  
figcolor=get(h_f,'color');  
edit2=uicontrol(h_f,'style','text','position',[100,330,400,40],'backgroundcolor',figcolor);  
button_open=uicontrol(h_f,'style','push','string','2.ѡ����Ƭ','position',[250 50 100 50],'callback','GuiOpen');  
button_recg=uicontrol(h_f,'style','push','string','1.��ʼ����','position',[100 50 100 50],'callback','GuiFace');  
button_match=uicontrol(h_f,'style','push','string','3.����ʶ��','position',[400 50 100 50],'callback','GuiRecognition'); 