FROM golang:1.14.1-alpine3.11
COPY ./source/main.go ./
RUN go build -o ./go-app ./main.go
USER nobody
ENTRYPOINT [ "./go-app" ]
