module MyFortranModule
implicit none 

contains

    subroutine zero_array_1D(ni, array) bind(C, name="zero_array_1D")
        integer,                 intent(in ) :: ni
        real(wp), dimension(ni), intent(out) :: array
        ! -----------------------
        integer :: i
        ! -----------------------
        do i = 1, ni
        array(i) = 0.0_wp
        end do
    end subroutine zero_array_1D

    subroutine add(a, b, result) bind(C, name="fortran_add")
    implicit none
    double precision, intent(in) :: a, b
    double precision, intent(out) :: result
    result = a + b
    end subroutine add

    subroutine hello_world() bind(C, name="fortran_hello_world")
        use iso_c_binding
        implicit none
        integer :: age = 99
        character(len = 20) :: greeting

        greeting = "Hello from test!"
        print *, greeting
        return
    end subroutine
end module MyFortranModule

program MyApp
use MyFortranModule
    implicit none
    double precision :: res, a=3.0, b=4.0
    call hello_world()
    call add(a, b, res)
    print *, res
end program
