#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# generated by wxGlade 0.9.9pre on Tue Jul 21 22:52:13 2020
#

import wx

# begin wxGlade: dependencies
# end wxGlade

# begin wxGlade: extracode
# end wxGlade


class MyDialog(wx.Dialog):
    def __init__(self, *args, **kwds):
        # begin wxGlade: MyDialog.__init__
        kwds["style"] = kwds.get("style", 0) | wx.DEFAULT_DIALOG_STYLE
        wx.Dialog.__init__(self, *args, **kwds)
        self.SetSize((400, 300))
        self.SetTitle("dialog")
        self.SetBackgroundColour(wx.Colour(75, 75, 75))

        sizer_1 = wx.BoxSizer(wx.VERTICAL)

        sizer_1.Add((20, 20), 0, 0, 0)

        static_text_1 = wx.StaticText(self, wx.ID_ANY, "Great !!")
        static_text_1.SetForegroundColour(wx.Colour(0, 255, 127))
        static_text_1.SetFont(wx.Font(16, wx.FONTFAMILY_DECORATIVE, wx.FONTSTYLE_SLANT, wx.FONTWEIGHT_BOLD, 0, ""))
        sizer_1.Add(static_text_1, 0, 0, 0)

        sizer_1.Add((20, 20), 0, 0, 0)

        static_text_2 = wx.StaticText(self, wx.ID_ANY, "Your result is Negative")
        static_text_2.SetForegroundColour(wx.Colour(0, 255, 127))
        static_text_2.SetFont(wx.Font(16, wx.FONTFAMILY_DECORATIVE, wx.FONTSTYLE_SLANT, wx.FONTWEIGHT_BOLD, 0, ""))
        sizer_1.Add(static_text_2, 0, 0, 0)

        sizer_1.Add((20, 20), 0, 0, 0)

        static_text_3 = wx.StaticText(self, wx.ID_ANY, "Stay Safe and Stay Healthy")
        static_text_3.SetForegroundColour(wx.Colour(0, 255, 127))
        static_text_3.SetFont(wx.Font(16, wx.FONTFAMILY_DECORATIVE, wx.FONTSTYLE_SLANT, wx.FONTWEIGHT_BOLD, 0, ""))
        sizer_1.Add(static_text_3, 0, 0, 0)

        sizer_1.Add((20, 20), 0, 0, 0)

        self.button_1 = wx.Button(self, wx.ID_ANY, "OK")
        self.button_1.SetFont(wx.Font(16, wx.FONTFAMILY_DECORATIVE, wx.FONTSTYLE_SLANT, wx.FONTWEIGHT_BOLD, 0, ""))
        sizer_1.Add(self.button_1, 0, wx.ALIGN_CENTER_HORIZONTAL, 0)

        self.SetSizer(sizer_1)

        self.Layout()

        self.Bind(wx.EVT_BUTTON, self.Ok_Button, self.button_1)
        # end wxGlade

    def Ok_Button(self, event):  # wxGlade: MyDialog.<event_handler>
        self.Close()

# end of class MyDialog

class MyApp(wx.App):
    def OnInit(self):
        self.dialog = MyDialog(None, wx.ID_ANY, "")
        self.SetTopWindow(self.dialog)
        self.dialog.ShowModal()
        self.dialog.Destroy()
        return True

# end of class MyApp

def run():
    app = MyApp(0)
    app.MainLoop()