#include <iostream>
#include <pthread.h>

void print_scope(std::ostream &ost, int scope)
{
  ost << "scope = ";
  switch (scope)
  {
  case PTHREAD_SCOPE_PROCESS:
    ost << "PTHREAD_SCOPE_PROCESS";
    break;
  case PTHREAD_SCOPE_SYSTEM:
    ost << "PTHREAD_SCOPE_SYSTEM";
    break;
  }
  ost << std::endl;
}

void print_inher(std::ostream &ost, int inher)
{
  ost << "inheritsched = ";
  switch (inher)
  {
  case PTHREAD_INHERIT_SCHED:
    ost << "PTHREAD_INHERIT_SCHED";
    break;
  case PTHREAD_EXPLICIT_SCHED:
    ost << "PTHREAD_EXPLICIT_SCHED";
    break;
  }
  ost << std::endl;
}

void print_detach(std::ostream &ost, int detach)
{
  ost << "detach = ";
  switch (detach)
  {
  case PTHREAD_CREATE_DETACHED:
    ost << "PTHREAD_CREATE_DETACHED";
    break;
  case PTHREAD_CREATE_JOINABLE:
    ost << "PTHREAD_CREATE_JOINABLE";
    break;
  }
  ost << std::endl;
}

void print_policy(std::ostream &ost, int policy)
{
  ost << "schedpolicy = ";
  switch (policy)
  {
  case SCHED_BATCH:
    ost << "SCHED_BATCH";
    break;
  case SCHED_DEADLINE:
    ost << "SCHED_DEADLINE";
    break;
  case SCHED_FIFO:
    ost << "SCHED_FIFO";
    break;
  case SCHED_IDLE:
    ost << "SCHED_IDLE";
    break;
  case SCHED_ISO:
    ost << "SCHED_ISO";
    break;
  case SCHED_OTHER:
    ost << "SCHED_OTHER";
    break;
  case SCHED_RESET_ON_FORK:
    ost << "SCHED_RESET_ON_FORK";
    break;
  case SCHED_RR:
    ost << "SCHED_RR";
    break;
  }
  ost << std::endl;
}



int main()
{
  pthread_attr_t attrs;

  pthread_attr_init(&attrs);
  int scope = 0;
  pthread_attr_getscope(&attrs, &scope);
  print_scope(std::cout, scope);

  int detach = 0;
  pthread_attr_getdetachstate(&attrs, &detach);
  print_detach(std::cout, detach);

  void *stk_ptr = 0;
  size_t stk_size = 0;
  pthread_attr_getstack(&attrs, &stk_ptr, &stk_size);
  std::cout << "stk_addr = " << reinterpret_cast<std::uintptr_t>(stk_ptr) << std::endl;
  std::cout << "stk_size = " << stk_size << std::endl;

  int inher = 0;
  pthread_attr_getinheritsched(&attrs, &inher);
  print_inher(std::cout, inher);

  int policy = 0;
  pthread_attr_getschedpolicy(&attrs, &policy);
  print_policy(std::cout, policy);

  pthread_attr_destroy(&attrs);
  return 0;
}