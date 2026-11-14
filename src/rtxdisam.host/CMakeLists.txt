cmake_minimum_required(VERSION 3.16)
project(rtxdisam_host C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_executable(rtxdisam_host rtxdisam_host.c)

if(MSVC)
  target_compile_options(rtxdisam_host PRIVATE /W3)
else()
  target_compile_options(rtxdisam_host PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(WIN32)
  target_link_libraries(rtxdisam_host PRIVATE ws2_32)
endif()

include(GNUInstallDirs)
install(TARGETS rtxdisam_host RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
