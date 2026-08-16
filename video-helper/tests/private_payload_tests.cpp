#include "PrivateInheritedPayload.h"
#include <array>
#include <cassert>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

int main()
{
    const auto readPacket=[](std::size_t bytes)
    {
        int fds[2]; assert(pipe(fds)==0);
        std::array<uint8_t,41> input{};
        assert(write(fds[1],input.data(),bytes)==ssize_t(bytes)); close(fds[1]);
        std::array<uint8_t,40> output{};
        return programmableruntime::privatepayload::readOneShot(fds[0],output,100);
    };
    assert(!readPacket(39));
    assert(readPacket(40));
    assert(!readPacket(41));

    int secret[2], output[2]; assert(pipe(secret)==0 && pipe(output)==0);
    for(int fd:{secret[0],secret[1],output[0],output[1]}) assert(fcntl(fd,F_SETFD,FD_CLOEXEC)==0);
    const pid_t pid=fork(); assert(pid>=0);
    if(pid==0)
    {
        close(secret[1]); close(output[0]);
        if(secret[0]!=3) { assert(dup2(secret[0],3)==3); close(secret[0]); }
        if(output[1]!=1) { assert(dup2(output[1],1)==1); close(output[1]); }
        assert(fcntl(3,F_SETFD,fcntl(3,F_GETFD)&~FD_CLOEXEC)==0);
        char* const args[]={const_cast<char*>(PRIVATE_PAYLOAD_FIXTURE),const_cast<char*>("fixture"),nullptr};
        execv(args[0],args); _exit(127);
    }
    close(secret[0]); close(output[1]);
    std::array<uint8_t,40> packet{}; for(size_t i=0;i<packet.size();++i) packet[i]=uint8_t(i+1);
    assert(write(secret[1],packet.data(),packet.size())==ssize_t(packet.size())); close(secret[1]);
    char buffer[256]{}; const auto count=read(output[0],buffer,sizeof(buffer)-1); close(output[0]);
    int status=0; assert(waitpid(pid,&status,0)==pid && WIFEXITED(status) && WEXITSTATUS(status)==0);
    const std::string report(buffer,count>0?size_t(count):0);
    assert(report=="bytes=40 closed=1 leaked=0 first=1 last=40\n");
}