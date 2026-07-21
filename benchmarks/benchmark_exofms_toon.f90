! Benchmark the Exo-FMS Toon shortwave and longwave solvers for independent
! atmospheric columns.  The solver modules are compiled from a separately
! checked-out Exo-FMS source tree; they are not vendored here.
program benchmark_exofms_toon
  use, intrinsic :: iso_fortran_env, only : real64
  use omp_lib, only : omp_get_max_threads, omp_get_wtime
  use sw_Toon_mod, only : sw_Toon
  use lw_Toon_mod, only : lw_Toon
  implicit none

  integer, parameter :: dp = real64
  integer :: nprofile, nlay, nlev, warmup, repeats
  integer :: profile, repeat, level, omp_threads
  real(dp) :: sw_seconds, lw_seconds, checksum, wall_start
  real(dp) :: f_inc, albedo, tint, contribution
  real(dp), allocatable :: tau_edge(:, :), mu_z(:, :), ssa(:, :), asymmetry(:, :)
  real(dp), allocatable :: temperature(:, :), pressure_layer(:, :), pressure_edge(:, :)
  real(dp), allocatable :: sw_up(:, :), sw_down(:, :), sw_net(:, :)
  real(dp), allocatable :: lw_up(:, :), lw_down(:, :), lw_net(:, :)
  character(len=32) :: argument

  call require_integer_argument(1, nprofile, 'nprofile')
  call optional_integer_argument(2, 40, nlay)
  call optional_integer_argument(3, 1, warmup)
  call optional_integer_argument(4, 3, repeats)
  if (nprofile < 1 .or. nlay < 4 .or. nlay > 150 .or. warmup < 0 .or. repeats < 1) then
    error stop 'usage: benchmark_exofms_toon nprofile [nlay=40] [warmup=1] [repeats=3]'
  end if

  nlev = nlay + 1
  allocate(tau_edge(nlev, nprofile), mu_z(nlev, nprofile), ssa(nlay, nprofile), asymmetry(nlay, nprofile))
  allocate(temperature(nlay, nprofile), pressure_layer(nlay, nprofile), pressure_edge(nlev, nprofile))
  allocate(sw_up(nlev, nprofile), sw_down(nlev, nprofile), sw_net(nlev, nprofile))
  allocate(lw_up(nlev, nprofile), lw_down(nlev, nprofile), lw_net(nlev, nprofile))

  do profile = 1, nprofile
    do level = 1, nlev
      tau_edge(level, profile) = 0.1_dp * real(level - 1, dp)
      mu_z(level, profile) = 0.5_dp
      pressure_edge(level, profile) = 1.0e2_dp * (1.0e3_dp ** (real(level - 1, dp) / real(nlay, dp)))
    end do
    do level = 1, nlay
      pressure_layer(level, profile) = sqrt(pressure_edge(level, profile) * pressure_edge(level + 1, profile))
      temperature(level, profile) = 300.0_dp - 0.5_dp * real(level - 1, dp)
    end do
  end do
  ssa = 0.5_dp
  asymmetry = 0.5_dp
  f_inc = 1.0_dp
  albedo = 0.1_dp
  tint = 100.0_dp
  omp_threads = omp_get_max_threads()

  checksum = 0.0_dp
  do repeat = 1, warmup
!$omp parallel do reduction(+:checksum) private(contribution) schedule(static)
    do profile = 1, nprofile
      call sw_contribution(nlay, nlev, tau_edge(:, profile), mu_z(:, profile), f_inc, ssa(:, profile), &
                           asymmetry(:, profile), albedo, sw_up(:, profile), sw_down(:, profile), &
                           sw_net(:, profile), contribution)
      checksum = checksum + contribution
      call lw_contribution(nlay, nlev, temperature(:, profile), pressure_layer(:, profile), &
                           pressure_edge(:, profile), tau_edge(:, profile), ssa(:, profile), &
                           asymmetry(:, profile), albedo, tint, lw_up(:, profile), lw_down(:, profile), &
                           lw_net(:, profile), contribution)
      checksum = checksum + contribution
    end do
!$omp end parallel do
  end do

  wall_start = omp_get_wtime()
  do repeat = 1, repeats
!$omp parallel do reduction(+:checksum) private(contribution) schedule(static)
    do profile = 1, nprofile
      call sw_contribution(nlay, nlev, tau_edge(:, profile), mu_z(:, profile), f_inc, ssa(:, profile), &
                           asymmetry(:, profile), albedo, sw_up(:, profile), sw_down(:, profile), &
                           sw_net(:, profile), contribution)
      checksum = checksum + contribution
    end do
!$omp end parallel do
  end do
  sw_seconds = (omp_get_wtime() - wall_start) / real(repeats, dp)

  wall_start = omp_get_wtime()
  do repeat = 1, repeats
!$omp parallel do reduction(+:checksum) private(contribution) schedule(static)
    do profile = 1, nprofile
      call lw_contribution(nlay, nlev, temperature(:, profile), pressure_layer(:, profile), &
                           pressure_edge(:, profile), tau_edge(:, profile), ssa(:, profile), &
                           asymmetry(:, profile), albedo, tint, lw_up(:, profile), lw_down(:, profile), &
                           lw_net(:, profile), contribution)
      checksum = checksum + contribution
    end do
!$omp end parallel do
  end do
  lw_seconds = (omp_get_wtime() - wall_start) / real(repeats, dp)

  write(*, '(a,i0,a,i0,a,i0,a,i0)') 'nprofile=', nprofile, ',nlay=', nlay, &
    ',warmup=', warmup, ',repeats=', repeats
  write(*, '(a,i0)') 'openmp_threads=', omp_threads
  write(*, '(a,f0.9)') 'exofms_sw_toon_seconds=', sw_seconds
  write(*, '(a,f0.9)') 'exofms_lw_toon_5node_seconds=', lw_seconds
  write(*, '(a,es24.16)') 'checksum=', checksum

contains

  subroutine require_integer_argument(position, value, name)
    integer, intent(in) :: position
    integer, intent(out) :: value
    character(len=*), intent(in) :: name

    if (command_argument_count() < position) error stop 'missing '//name
    call get_command_argument(position, argument)
    read(argument, *) value
  end subroutine require_integer_argument

  subroutine optional_integer_argument(position, default_value, value)
    integer, intent(in) :: position, default_value
    integer, intent(out) :: value

    value = default_value
    if (command_argument_count() >= position) then
      call get_command_argument(position, argument)
      read(argument, *) value
    end if
  end subroutine optional_integer_argument

  subroutine sw_contribution(nlay, nlev, tau_edge, mu_z, f_inc, ssa, asymmetry, albedo, &
                             sw_up, sw_down, sw_net, result)
    integer, intent(in) :: nlay, nlev
    real(dp), intent(in) :: tau_edge(nlev), mu_z(nlev), f_inc, ssa(nlay), asymmetry(nlay), albedo
    real(dp), intent(out) :: sw_up(nlev), sw_down(nlev), sw_net(nlev), result
    real(dp) :: mu_z_local(nlev), asr

    mu_z_local = mu_z
    call sw_Toon(nlay, nlev, tau_edge, mu_z_local, f_inc, ssa, asymmetry, albedo, &
                 sw_up, sw_down, sw_net, asr)
    result = sw_net(nlev) + asr
  end subroutine sw_contribution

  subroutine lw_contribution(nlay, nlev, temperature, pressure_layer, pressure_edge, tau_edge, &
                             ssa, asymmetry, albedo, tint, lw_up, lw_down, lw_net, result)
    integer, intent(in) :: nlay, nlev
    real(dp), intent(in) :: temperature(nlay), pressure_layer(nlay), pressure_edge(nlev), tau_edge(nlev)
    real(dp), intent(in) :: ssa(nlay), asymmetry(nlay), albedo, tint
    real(dp), intent(out) :: lw_up(nlev), lw_down(nlev), lw_net(nlev), result
    real(dp) :: olr

    call lw_Toon(nlay, nlev, temperature, pressure_layer, pressure_edge, tau_edge, ssa, asymmetry, &
                 albedo, tint, lw_up, lw_down, lw_net, olr)
    result = lw_net(1) + olr
  end subroutine lw_contribution

end program benchmark_exofms_toon
