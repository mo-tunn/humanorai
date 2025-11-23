Public Class Form1

#Region "Snake Staff"
    Dim snake(1000) As PictureBox
    Dim lenght_of_Snake As Integer = -1
    Dim left_right_mover As Integer = 0
    Dim up_down_mover As Integer = 0
    Dim r As New Random
    Dim Mouse = New PictureBox


    Private Sub create_head()
        lenght_of_Snake += 1
        snake(lenght_of_Snake) = New PictureBox
        With snake(lenght_of_Snake)
            .Height = 10
            .Width = 10
            .BackColor = Color.White
            .Top = (pb_Field.Top + pb_Field.Bottom) / 2
            .Left = (pb_Field.Left + pb_Field.Right) / 2
        End With
        Me.Controls.Add(snake(lenght_of_Snake))
        snake(lenght_of_Snake).BringToFront()
        lenght_Snake()
    End Sub
    Private Sub lenght_Snake()
        lenght_of_Snake += 1
        snake(lenght_of_Snake) = New PictureBox
        With snake(lenght_of_Snake)
            .Height = 10
            .Width = 10
            .BackColor = Color.White
            .Top = snake(lenght_of_Snake - 1).Top
            .Left = snake(lenght_of_Snake - 1).Left
        End With
        Me.Controls.Add(snake(lenght_of_Snake - 1))
        snake(lenght_of_Snake).BringToFront()
    End Sub
    Private Sub Form1_KeyPress(ByVal sender As Object, ByVal e As System.Windows.Forms.KeyPressEventArgs) Handles Me.KeyPress
        tm_SnakeMover.Start()
        Select Case e.KeyChar
            Case "a"
                left_right_mover = -10
                up_down_mover = 0
            Case "d"
                left_right_mover = 10
                up_down_mover = 0
            Case "w"
                left_right_mover = 0
                up_down_mover = -10
            Case "s"
                left_right_mover = 0
                up_down_mover = 10
        End Select

    End Sub
    Private Sub tm_SnakeMover_Tick(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles tm_SnakeMover.Tick
        For i = lenght_of_Snake To 1 Step -1
            snake(i).Top = snake(i - 1).Top
            snake(i).Left = snake(i - 1).Left
        Next

        snake(0).Top += up_down_mover
        snake(0).Left += left_right_mover
        collide_With_walls()
        collide_with_mouse()
        collide_with_itself()
    End Sub
#End Region
    Private Sub Form1_Load(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles MyBase.Load
        create_head()
        Create_mouse()
        'tm_SnakeMover.Start()
    End Sub
#Region "Collation"
    Private Sub collide_With_walls()
        If snake(0).left < pb_Field.left Then
            tm_SnakeMover.Stop()
            MsgBox("You Lost")
        End If
        If snake(0).Right > pb_Field.Right Then
            tm_SnakeMover.Stop()
            MsgBox("You Lost")
        End If
        If snake(0).Top < pb_Field.Top Then
            tm_SnakeMover.Stop()
            MsgBox("You Lost")
        End If
        If snake(0).Bottom > pb_Field.Bottom Then
            tm_SnakeMover.Stop()
            MsgBox("You Lost")
        End If
    End Sub
    Private Sub collide_with_mouse()
        If snake(0).Bounds.IntersectsWith(Mouse.bounds) Then
            lenght_Snake()
            Mouse.Left = r.Next(pb_Field.Left, pb_Field.Right - 10)
            Mouse.Top = r.Next(pb_Field.Top, pb_Field.Bottom - 10)
        End If
    End Sub
    Private Sub collide_with_itself()
        For i = 1 To lenght_of_Snake - 1
            If snake(0).Bounds.IntersectsWith(snake(i).Bounds) Then
                tm_SnakeMover.Stop()
                MsgBox("You Lost Lose")

            End If
        Next
    End Sub
#End Region
#Region "Mouse Staff"
    Private Sub Create_mouse()
        With Mouse
            .Width = 10
            .Height = 10
            .BackColor = Color.Red
            .Left = r.Next(pb_Field.Left, pb_Field.Right - 10)
            .Top = r.Next(pb_Field.Top, pb_Field.Bottom - 10)
        End With
        Me.Controls.Add(Mouse)
        Mouse.BringToFront()
    End Sub
#End Region

    Private Sub pb_Field_Click(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles pb_Field.Click

    End Sub
End Class
