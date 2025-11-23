<Global.Microsoft.VisualBasic.CompilerServices.DesignerGenerated()> _
Partial Class Form1
    Inherits System.Windows.Forms.Form

    'Form overrides dispose to clean up the component list.
    <System.Diagnostics.DebuggerNonUserCode()> _
    Protected Overrides Sub Dispose(ByVal disposing As Boolean)
        Try
            If disposing AndAlso components IsNot Nothing Then
                components.Dispose()
            End If
        Finally
            MyBase.Dispose(disposing)
        End Try
    End Sub

    'Required by the Windows Form Designer
    Private components As System.ComponentModel.IContainer

    'NOTE: The following procedure is required by the Windows Form Designer
    'It can be modified using the Windows Form Designer.  
    'Do not modify it using the code editor.
    <System.Diagnostics.DebuggerStepThrough()> _
    Private Sub InitializeComponent()
        Me.components = New System.ComponentModel.Container
        Me.pb_Field = New System.Windows.Forms.PictureBox
        Me.tm_SnakeMover = New System.Windows.Forms.Timer(Me.components)
        CType(Me.pb_Field, System.ComponentModel.ISupportInitialize).BeginInit()
        Me.SuspendLayout()
        '
        'pb_Field
        '
        Me.pb_Field.BackColor = System.Drawing.SystemColors.GradientActiveCaption
        Me.pb_Field.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle
        Me.pb_Field.Location = New System.Drawing.Point(12, 16)
        Me.pb_Field.Name = "pb_Field"
        Me.pb_Field.Size = New System.Drawing.Size(620, 454)
        Me.pb_Field.TabIndex = 0
        Me.pb_Field.TabStop = False
        Me.pb_Field.Visible = False
        '
        'tm_SnakeMover
        '
        Me.tm_SnakeMover.Enabled = True
        Me.tm_SnakeMover.Interval = 50
        '
        'Form1
        '
        Me.AutoScaleDimensions = New System.Drawing.SizeF(6.0!, 13.0!)
        Me.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font
        Me.BackColor = System.Drawing.SystemColors.GradientActiveCaption
        Me.ClientSize = New System.Drawing.Size(644, 482)
        Me.Controls.Add(Me.pb_Field)
        Me.Name = "Form1"
        Me.Text = "Form1"
        CType(Me.pb_Field, System.ComponentModel.ISupportInitialize).EndInit()
        Me.ResumeLayout(False)

    End Sub
    Friend WithEvents pb_Field As System.Windows.Forms.PictureBox
    Friend WithEvents tm_SnakeMover As System.Windows.Forms.Timer

End Class
