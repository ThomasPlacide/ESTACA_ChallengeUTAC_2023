#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <pthread.h>

#define HEADER 64
#define PORT 12345
#define FORMAT "utf-8"
#define DISCONNECT_MESSAGE "!DISCONNECT"

void *handle_client(void *arg) {
    int client_socket = *(int *)arg;
    char buffer[HEADER] = {0};
    printf("[NOUVELLE CONNEXION] Client connecté.\n");
    int connected = 1;

    while (connected) {
        int msg_length = 0;
        if (recv(client_socket, &msg_length, sizeof(msg_length), 0) > 0) {
            msg_length = ntohl(msg_length);
            char *msg = (char *)malloc(msg_length + 1);
            memset(msg, 0, msg_length + 1);
            if (recv(client_socket, msg, msg_length, 0) > 0) {
                if (strcmp(msg, DISCONNECT_MESSAGE) == 0) {
                    connected = 0;
                }
                printf("[%d] %s\n", client_socket, msg);
                send(client_socket, "Message reçu", strlen("Message reçu"), 0);
            }
            free(msg);
        }
    }
    close(client_socket);
    printf("[DÉCONNEXION] Client déconnecté.\n");
    return NULL;
}

void start() {
    int server_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (server_socket < 0) {
        perror("Erreur lors de la création de la socket");
        exit(EXIT_FAILURE);
    }

    int opt = 1;
    if (setsockopt(server_socket, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt)) < 0) {
        perror("Erreur lors du paramétrage de la socket");
        exit(EXIT_FAILURE);
    }

    struct sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);

    if (bind(server_socket, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("Erreur lors du binding de la socket");
        exit(EXIT_FAILURE);
    }

    if (listen(server_socket, SOMAXCONN) < 0) {
        perror("Erreur lors de l'écoute de la socket");
        exit(EXIT_FAILURE);
    }

    printf("[EN ÉCOUTE] Le serveur est en écoute sur le port %d...\n", PORT);

    int client_socket;
    struct sockaddr_in client_address;
    socklen_t client_address_len = sizeof(client_address);
    pthread_t thread_id;

    while ((client_socket = accept(server_socket, (struct sockaddr *)&client_address, &client_address_len))) {
        if (pthread_create(&thread_id, NULL, handle_client, &client_socket) != 0) {
            perror("Erreur lors de la création du thread");
            exit(EXIT_FAILURE);
        }
        printf("[CONNEXIONS ACTIVES] %d\n", (int)pthread_self());
    }

    close(server_socket);
}

int main() {
    printf("[DÉMARRAGE] Le serveur est en train de démarrer...\n");
    start();
    return 0;
}

