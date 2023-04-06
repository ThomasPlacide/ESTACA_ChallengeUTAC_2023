#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <time.h>

#define SERVER_ADDRESS "193.169.0.6"  // Adresse IP du serveur
#define SERVER_PORT 12345             // Port du serveur

int main() {
    int client_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (client_socket < 0) {
        perror("Erreur lors de la création de la socket");
        exit(EXIT_FAILURE);
    }

    struct sockaddr_in server_address;
    server_address.sin_family = AF_INET;
    server_address.sin_addr.s_addr = inet_addr(SERVER_ADDRESS);
    server_address.sin_port = htons(SERVER_PORT);

    if (connect(client_socket, (struct sockaddr *)&server_address, sizeof(server_address)) < 0) {
        perror("Erreur lors de la connexion au serveur");
        exit(EXIT_FAILURE);
    }

    time_t current_time = time(NULL);
    struct tm *time_info = localtime(&current_time);
    char time_str[80];
    strftime(time_str, sizeof(time_str), "%A %d %B %Y %H:%M:%S", time_info);

    char message[1024];
    snprintf(message, sizeof(message), "Date : %s\nMessage : Hello world", time_str);

    if (send(client_socket, message, strlen(message), 0) < 0) {
        perror("Erreur lors de l'envoi du message");
        exit(EXIT_FAILURE);
    }

    printf("Message envoyé :\n%s\n", message);

    close(client_socket);
    return 0;
}

