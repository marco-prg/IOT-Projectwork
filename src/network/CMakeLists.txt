add_executable(main main.cpp ${tiny_dnn_headers})
target_link_libraries(main
        ${project_library_target_name} ${REQUIRED_LIBRARIES})

cotire(main)